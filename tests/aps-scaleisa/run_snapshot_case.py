#!/usr/bin/env python3
"""Validate repository PCL/GTSAM snapshots with an independent C++ reference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from run_octree_case import ROOT, DRIVER, _extract_function_blocks


CASES = {
    'ransac': ('pcl/sample_consensus_ransac', {'POINTS': 32, 'HYPOTHESES': 4}),
    'segmentation': ('pcl/segmentation_region_growing', {'HEIGHT': 8, 'WIDTH': 8, 'ITERS': 2}),
    'poisson': ('pcl/surface_poisson', {'SAMPLES': 24, 'GRID': 6, 'ITERS': 2}),
    **{name: ('gtsam/' + name, {}) for name in
       ('sfm_batch', 'stereo_vo', 'planar_slam', 'pose2_slam', 'pose3_pose_graph')},
}


def prepare(case: str, directory: Path) -> tuple[Path, Path, int, list[int]]:
    origin, dimensions = CASES[case]
    base = ROOT / 'scaleisa/benchmarks' / origin
    original_cpp = (base / (base.name + '.cpp')).read_text()
    original_mlir = (base / (base.name + '.mlir')).read_text()
    cpp, mlir = original_cpp, original_mlir
    for name, value in dimensions.items():
        cpp, changes = re.subn(r'(static const int ' + name + r' = )\d+;',
                               rf'\g<1>{value};', cpp)
        if changes != 1:
            raise RuntimeError(f'Missing dimension {name}')
    replacements = {
        'ransac': {'1024x': '32x', '96x': '4x', 'to 1024 {': 'to 32 {',
                   'to 96 {': 'to 4 {', 'constant 1024 : i32': 'constant 32 : i32'},
        'segmentation': {'72x': '8x', '96x': '8x', 'PA96_': 'PA8_',
                         'to 72 {': 'to 8 {', 'to 96 {': 'to 8 {', 'to 8 {': 'to 2 {',
                         'constant 71 : i32': 'constant 7 : i32',
                         'constant 95 : i32': 'constant 7 : i32',
                         'constant 96 : i32': 'constant 8 : i32'},
        'poisson': {'768x': '24x', '13824x': '216x', 'to 768 {': 'to 24 {',
                    'to 13824 {': 'to 216 {', 'to 22 {': 'to 4 {',
                    'to 12 {': 'to 2 {', 'constant 24 : i32': 'constant 6 : i32',
                    'constant 23 : i32': 'constant 5 : i32',
                    'constant 2.400000e+01 : f32': 'constant 6.000000e+00 : f32'},
    }.get(case, {})
    if replacements:
        # Simultaneous substitution prevents new dimensions matching old bounds.
        mlir = re.sub('|'.join(map(re.escape, sorted(replacements, key=len, reverse=True))),
                      lambda m: replacements[m[0]], mlir)
    constants = {}
    for name, expression in re.findall(r'static const int (\w+) = ([^;]+);', cpp):
        for known, value in constants.items():
            expression = re.sub(r'\b' + known + r'\b', str(value), expression)
        if not re.fullmatch(r'[\d *+()-]+', expression):
            raise RuntimeError(f'Unsupported constant expression {expression}')
        constants[name] = eval(expression, {'__builtins__': {}}, {})
    cpp_main = cpp.split('int main() {')[1]
    arrays = re.findall(r'^  (float|int) (\w+)((?:\[[\w]+\])+);', cpp_main, re.MULTILINE)
    main = _extract_function_blocks(mlir)['main']
    allocations = re.findall(r'(%[\w.$-]+) = memref.alloca\(\) : (memref<[^>]+>)', main)
    ordered = list(reversed(allocations))
    if len(arrays) != len(ordered):
        raise RuntimeError('Snapshot entry allocations changed')
    counts, types, arguments, copies, checks = [], [], [], [], []
    initial_copies, global_declarations, global_definitions, staging = [], [], [], []
    solver_only = case in {'pose2_slam', 'pose3_pose_graph'}
    for i, ((ctype, name, shape), (value, typ)) in enumerate(zip(arrays, ordered)):
        sizes = [int(x) if x.isdecimal() else constants[x] for x in re.findall(r'\[(\w+)\]', shape)]
        elem = 'f32' if ctype == 'float' else 'i32'
        expected = 'memref<' + 'x'.join(map(str, sizes)) + 'x' + elem + '>'
        if typ != expected:
            raise RuntimeError(f'Allocation mismatch for {name}: {typ} != {expected}')
        count = math.prod(sizes)
        counts.append(count)
        types.append(typ)
        arguments.append(value)
        copies.append(f'  __builtin_memcpy(expected_{i}, {name}, sizeof(expected_{i}));')
        checks.append(f'  check_words(buffer_{i}, expected_{i}, {count});')
        if solver_only:
            global_name = f'bench_input_{i}'
            global_declarations.append(f'extern "C" {ctype} {global_name}[{count}];')
            global_definitions.append(f'  memref.global @{global_name} : {typ} = uninitialized')
            initial_copies.append(f'  __builtin_memcpy({global_name}, {name}, sizeof({global_name}));')
            prefix = [f'    %initial_{i}=memref.get_global @{global_name}:{typ}']
            indices = ','.join(f'%stage_{i}_{d}' for d in range(len(sizes)))
            prefix.extend(f'affine.for %stage_{i}_{d}=0 to {extent} {{' for d, extent in enumerate(sizes))
            prefix.extend([f'%initial_value_{i}=affine.load %initial_{i}[{indices}]:{typ}',
                           f'affine.store %initial_value_{i},{value}[{indices}]:{typ}'])
            prefix.extend('}' for _ in sizes)
            staging.extend(prefix)
    total = sum(counts)
    prepare_cpp = ''
    if solver_only:
        solver = 'solve_pose_graph' if case == 'pose2_slam' else 'solve_pose3'
        prepare_cpp = ('static void prepare_inputs() {' + cpp_main.split('  ' + solver + '(')[0]
                       + '\n'.join(initial_copies) + '\n}\n')
        initializers = ('build_pose_graph', 'init_state') if case == 'pose2_slam' else ('init_pose3_problem',)
        replacements = 0
        def replace_initializer(match):
            nonlocal replacements
            replacements += 1
            return '\n'.join(staging) if replacements == 1 else ''
        pattern = r'(?m)^\s*call @[^\n]*(?:' + '|'.join(initializers) + r')[^\n]*$'
        main = re.sub(pattern, replace_initializer, main)
        if replacements != len(initializers):
            raise RuntimeError('Expected graph construction calls before solver')
        old_main = _extract_function_blocks(mlir)['main']
        mlir = mlir.replace(old_main, main, 1)
        mlir = mlir.replace('module {', 'module {\n' + '\n'.join(global_definitions), 1)
    cpp_main, changes = re.subn(r'(?m)^(  return [^\n]+)$',
                               lambda m: '\n'.join(copies) + '\n' + m[0], cpp_main)
    if changes != 1:
        raise RuntimeError('Expected one reference return')
    cpp = cpp.split('int main() {')[0] + 'int reference_main() {' + cpp_main
    declarations = '\n'.join(f'static uint32_t expected_{i}[{n}];' for i, n in enumerate(counts))
    signature = ', '.join(f'const void *buffer_{i}' for i in range(len(arrays)))
    harness = f'''#include <stdint.h>
{declarations}
extern "C" void *memcpy(void *dest, const void *source, __SIZE_TYPE__ n) {{
  volatile unsigned char *d=(volatile unsigned char *)dest;
  const volatile unsigned char *s=(const volatile unsigned char *)source;
  for(__SIZE_TYPE__ i=0;i<n;++i)d[i]=s[i];
  return dest;
}}
#include "reference.cpp"
{chr(10).join(global_declarations)}
{prepare_cpp}
volatile int test_status, checked_outputs, validation_calls, mismatches;
volatile int first_mismatch = -1;
volatile uint32_t first_actual, first_expected;
extern "C" int pcl_selected_main();
extern "C" volatile uint64_t tohost, fromhost;
static void report_mismatch() {{
  char text[128]; int n=0;
  const char *label="APS_MISMATCH index actual expected checked (hex): ";
  while(*label)text[n++]=*label++;
  uint32_t fields[4]={{(uint32_t)first_mismatch,first_actual,first_expected,(uint32_t)checked_outputs}};
  for(int j=0;j<4;++j) {{
    for(int shift=28;shift>=0;shift-=4) {{
      unsigned digit=(fields[j]>>shift)&15;
      text[n++]=digit<10?'0'+digit:'a'+digit-10;
    }}
    text[n++]=j==3?'\\n':' ';
  }}
  alignas(64) volatile uint64_t request[8]={{64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0}};
  asm volatile("fence rw,rw" ::: "memory");
  tohost=(uintptr_t)request;
  while(!fromhost){{}}
  fromhost=0;
}}
static void check_words(const void *actual, const uint32_t *expected, int n) {{
  for(int i=0;i<n;++i) {{
    uint32_t value;
    __builtin_memcpy(&value, (const char *)actual+4*i, 4);
    if(value!=expected[i]) {{
      if(first_mismatch<0) {{ first_mismatch=checked_outputs; first_actual=value; first_expected=expected[i]; }}
      ++mismatches;
    }}
    ++checked_outputs;
  }}
}}
extern "C" void pcl_validate_outputs({signature}) {{
  ++validation_calls;
{chr(10).join(checks)}
}}
extern "C" int main() {{
  int expected=reference_main();
  {'prepare_inputs();' if solver_only else ''}
  int actual=pcl_selected_main();
  if(validation_calls!=1 || checked_outputs!={total}) return 3;
  if(mismatches) {{report_mismatch(); return 1;}}
  if(actual!=expected) return 2;
  return 0;
}}
'''
    callback = ('    func.call @pcl_validate_outputs(' + ', '.join(arguments) + ') : ('
                + ', '.join(types) + ') -> ()\n')
    observed, changes = re.subn(r'(?m)^(\s*return [^\n]+)$', lambda m: callback + m[0], main)
    if changes != 1:
        raise RuntimeError('Expected one MLIR entry return')
    observed = mlir.replace(main, observed, 1).replace('@main(', '@pcl_selected_main(', 1)
    head, tail = observed.rsplit('}', 1)
    observed = head + '  func.func private @pcl_validate_outputs(' + ', '.join(types) + ')\n}' + tail
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'input.mlir').write_text(mlir)
    (directory / 'observed-baseline.mlir').write_text(observed)
    (directory / 'reference.cpp').write_text(cpp)
    (directory / 'main.cpp').write_text(harness)
    (directory / 'provenance.json').write_text(json.dumps({
        'origin': origin, 'dimensions': constants, 'overridden_dimensions': dimensions,
        'original_cpp_sha256': hashlib.sha256(original_cpp.encode()).hexdigest(),
        'original_mlir_sha256': hashlib.sha256(original_mlir.encode()).hexdigest(),
        'input_sha256': hashlib.sha256(mlir.encode()).hexdigest(),
        'observed_allocations': ordered, 'checked_values': total,
        'reference': 'Original C++ computation; capture all entry arrays before returning',
        'timed_scope': 'Solver and bounded input staging; graph construction outside ROI' if solver_only else 'Complete repository snapshot main',
    }, indent=2) + '\n')
    return directory / 'input.mlir', directory / 'main.cpp', total, list(reversed(range(len(arrays))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-pipeline', action='store_true')
    parser.add_argument('--discovery-dir', type=Path)
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    source, harness, values, indices = prepare(args.case, out / 'input')
    if args.prepare_only:
        return
    prefix = ROOT / '.pixi/envs/default'
    env = {**os.environ, 'PATH': f'{prefix}/bin:{prefix}/riscv-tools/bin:' + os.environ['PATH'],
           'LD_LIBRARY_PATH': str(prefix / 'lib')}

    def run(stage: str, command: list) -> None:
        print(args.case + ': ' + stage, flush=True)
        with (out / (stage + '.log')).open('w') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=7200)

    link = ['--emit=elf', '--entry=_start', '--abi=ilp32f',
            '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--link-source={harness}',
            f'--link-source={ROOT}/tests/scaleisa-compiler/fixtures/start.S',
            f'--link-arg=-T{ROOT}/tests/scaleisa-compiler/fixtures/ram.ld']
    run('normalize-baseline', [ROOT / 'build/tools/scaleisa-opt/scaleisa-opt',
        out / 'input/observed-baseline.mlir', '--allow-unregistered-dialect',
        '--scaleisa-egraph-preprocess', '-o', out / 'baseline.mlir'])
    empty = out / 'baseline-hardware.mlir'
    empty.write_text('module {}\n')
    run('compile-baseline', [DRIVER, out / 'baseline.mlir', '--compile-only',
        f'--hardware={empty}', '--allow-no-ci', *link, '-o', out / 'baseline.elf'])
    assert json.loads((out / 'baseline.elf.scaleisa/compile-report.json').read_text())['ci_sites'] == 0
    run('baseline-simulation', [prefix / 'riscv-tools/bin/spike',
        '--isa=RV32IMAFDC_Zicsr_Zifencei', out / 'baseline.elf'])
    if not args.reuse_pipeline:
        run('pipeline', [DRIVER, source, '--area-budget=10000', '--clock=8',
            '--max-extra-csrs=8', f'--output-dir={out}', '--cosim', f'--jobs={args.jobs}',
            *([f'--discovery-dir={args.discovery_dir.resolve()}'] if args.discovery_dir else []),
            '--rename-main=pcl_selected_main', '--observe-function=pcl_validate_outputs',
            *[f'--observe-allocation={i}' for i in indices], *link])
    report = json.loads((out / 'report.json').read_text())
    assert report['status'] == 'passed'
    assert report.get('cosimulation') == 'passed' or report.get('software_simulation') == 'passed'
    assert report['input_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    selection = report['selection']
    summary = dict(status='passed', scope='Repository ' + CASES[args.case][0] + ' snapshot',
        independently_checked_values_per_binary=values, selected_correctness='passed',
        normalized_software_baseline='passed; zero CIs', area_budget=10000,
        scheduling_period_ns=8, max_extra_csrs=8,
        selected_area_proxy=selection['selected_area'],
        software_model_cycles=selection['software_only_latency'],
        selected_model_cycles=selection['selected_latency'],
        estimated_program_speedup=selection['software_only_latency'] / selection['selected_latency'],
        selected_implementations=selection['selected_impls'], ci_sites=report['compiled']['ci_sites'],
        ci_cycle_comparison=report.get('ci_cycle_comparison', []),
        backend_pipeline_fallback_count=report.get('backend_pipeline_fallback_count'),
        validation_harness=str(harness), pipeline_report=str(out / 'report.json'),
        input_sha256=report['input_sha256'], target_profile_sha256=report['target_profile_sha256'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
