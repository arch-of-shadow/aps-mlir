#!/usr/bin/env python3
"""Validate small repository PCL snapshots through automatic AU selection."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from run_octree_case import ROOT, DRIVER, _extract_function_blocks


def prepare_input(directory: Path, case: str, size: int, points: int = 32,
                  queries: int = 4, iterations: int = 4) -> tuple[Path, Path, int]:
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT/'scaleisa/benchmarks/pcl'/case
    original_cpp = (source/f'{case}.cpp').read_text()
    original_mlir = (source/f'{case}.mlir').read_text()
    reference, count = re.subn(r'(static const int (?:WIDTH|HEIGHT) = )\d+;',
                              rf'\g<1>{size};', original_cpp)
    if count != 2 and case in {'normal_estimation', 'morphological_filter'}:
        raise RuntimeError('Source dimensions changed')
    text = original_mlir
    dimensions = [size, size]
    if case == 'normal_estimation':
        replacements = [('40x', f'{size}x'), ('48x', f'{size}x'),
                        ('PA48_', f'PA{size}_'), ('to 40 {', f'to {size} {{'),
                        ('to 48 {', f'to {size} {{'), ('to 38 {', f'to {size-2} {{'),
                        ('to 46 {', f'to {size-2} {{'),
                        ('constant 39 : i32', f'constant {size-1} : i32'),
                        ('constant 47 : i32', f'constant {size-1} : i32')]
        buffers = [('%alloca_6', f'memref<{size}x{size}x3xf32>'),
                   ('%alloca_5', f'memref<{size}x{size}x3xf32>'),
                   ('%alloca_4', f'memref<{size}x{size}x3xf32>'),
                   ('%alloca', f'memref<{size}x{size}xf32>')]
        fixture, values = 'normal_main.cpp', size*size*10
    elif case == 'morphological_filter':
        replacements = [('96x', f'{size}x'), ('PA96_', f'PA{size}_'),
                        ('to 96 {', f'to {size} {{'),
                        ('constant 95 : i32', f'constant {size-1} : i32')]
        buffers = [(name, f'memref<{size}x{size}xf32>')
                   for name in ('%alloca_2', '%alloca_1', '%alloca_0', '%alloca')]
        fixture, values = 'morphological_main.cpp', size*size*4
    elif case == 'kdtree_search':
        reference = original_cpp.replace('POINTS = 1536;', f'POINTS = {points};').replace(
            'QUERIES = 96;', f'QUERIES = {queries};')
        replacements = [('1536x', f'{points}x'), ('96x', f'{queries}x'),
                        ('to 1536 {', f'to {points} {{'), ('to 96 {', f'to {queries} {{')]
        buffers = [('%alloca_4', f'memref<{points}x3xf32>'),
                   ('%alloca_3', f'memref<{queries}x3xf32>'),
                   ('%alloca_2', f'memref<{queries}x8xf32>'),
                   ('%alloca', f'memref<{queries}x8xi32>')]
        fixture, values = 'knn_main.cpp', points*3+queries*19
        dimensions = [points, queries, 8]
    elif case == 'registration_icp':
        reference = original_cpp.replace('POINTS = 256;', f'POINTS = {points};').replace(
            'ITERS = 12;', f'ITERS = {iterations};')
        replacements = [('256x', f'{points}x'), ('12xf32', f'{iterations}xf32'),
                        ('to 256 {', f'to {points} {{'), ('to 12 {', f'to {iterations} {{'),
                        ('2.560000e+02', f'{float(points):.6e}')]
        buffers = [('%alloca_2', f'memref<{points}x3xf32>'),
                   ('%alloca_1', f'memref<{points}x3xf32>'),
                   ('%alloca_0', 'memref<3xf32>'),
                   ('%alloca', f'memref<{iterations}xf32>')]
        fixture, values = 'icp_main.cpp', points*6+3+iterations
        dimensions = [points, iterations]
    else:
        raise ValueError(f'Unknown PCL case: {case}')
    for old, new in replacements:
        text = text.replace(old, new)
    computation = text
    # Repeated buffer types require explicit snapshot allocation identities.
    # This callback observes outputs only; the original computations stay intact.
    main = _extract_function_blocks(text)['main']
    for name, typ in buffers:
        if not re.search(re.escape(name) + r' = memref\.alloca\(\) : ' + re.escape(typ), main):
            raise RuntimeError(f'Expected snapshot allocation {name}: {typ}')
    types = ', '.join(typ for _, typ in buffers)
    callback = ('    func.call @pcl_validate_outputs(' + ', '.join(name for name, _ in buffers)
                + ') : (' + types + ') -> ()\n')
    observed, count = re.subn(r'(?m)^(\s*return [^\n]+)$', lambda m: callback+m[0], main)
    if count != 1:
        raise RuntimeError('Expected one main return')
    text = text.replace(main, observed, 1)
    head, tail = text.rsplit('}', 1)
    text = head + f'  func.func private @pcl_validate_outputs({types})\n}}' + tail
    path = directory/f'{case}.mlir'
    path.write_text(computation)
    (directory/'observed-baseline.mlir').write_text(
        text.replace('@main(', '@pcl_selected_main(', 1))
    (directory/'reference.cpp').write_text(reference)
    harness = directory/'main.cpp'
    harness.write_text((ROOT/'tests/aps-scaleisa/fixtures'/fixture).read_text())
    (directory/'provenance.json').write_text(json.dumps({
        'case': case, 'dimensions': dimensions,
        'transformations': ['Spatial dimensions, matching type/loop bounds, clamps and symbols',
                            'Output observer added after selection; kernels unchanged'],
        'original_source_sha256': hashlib.sha256(original_cpp.encode()).hexdigest(),
        'original_mlir_sha256': hashlib.sha256(original_mlir.encode()).hexdigest(),
        'input_sha256': hashlib.sha256(computation.encode()).hexdigest(),
        'scaled_source_sha256': hashlib.sha256(reference.encode()).hexdigest(),
        'observed_allocations': buffers}, indent=2)+'\n')
    return path, harness, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['normal_estimation', 'morphological_filter',
                                        'kdtree_search', 'registration_icp'])
    parser.add_argument('--size', type=int, default=12)
    parser.add_argument('--points', type=int, default=32)
    parser.add_argument('--queries', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--max-extra-csrs', type=int, default=5)
    parser.add_argument('--area-budget', type=int, default=10000)
    parser.add_argument('--clock', type=float, default=8)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-pipeline', action='store_true')
    parser.add_argument('--discovery-dir', type=Path,
                        help='Reuse matching evaluated discovery when rebuilding downstream stages')
    args = parser.parse_args()
    if args.size < 5:
        parser.error('size must accommodate the complete five-point stencil')
    if args.points < 8 or args.queries < 1 or args.iterations < 1:
        parser.error('need at least eight points, one query and one iteration')
    out = args.output_dir.resolve()
    source, harness, values = prepare_input(out/'input', args.case, args.size,
                                          args.points, args.queries, args.iterations)
    prefix = ROOT/'.pixi/envs/default'
    env = os.environ.copy()
    env['PATH'] = f'{prefix}/bin:{prefix}/riscv-tools/bin:{env.get("PATH", "")}'
    env['LD_LIBRARY_PATH'] = str(prefix/'lib')

    def run(stage: str, command: list) -> None:
        print(stage, flush=True)
        with (out/(stage+'.log')).open('w') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)

    run('parse-input', [ROOT/'circt/build/bin/mlir-opt', '--allow-unregistered-dialect', source, '-o', os.devnull])
    link = ['--emit=elf', '--entry=_start', '--abi=ilp32f',
            '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--link-source={harness}',
            f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
            f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}']
    if not args.reuse_pipeline:
        run('pipeline', [DRIVER, source, f'--area-budget={args.area_budget}',
            f'--output-dir={out}', f'--clock={args.clock}', '--cosim',
            f'--max-extra-csrs={args.max_extra_csrs}',
            *([f'--discovery-dir={args.discovery_dir.resolve()}'] if args.discovery_dir else []),
            '--rename-main=pcl_selected_main', '--observe-function=pcl_validate_outputs',
            *[f'--observe-allocation={index}' for index in (3, 2, 1, 0)], *link])
    report = json.loads((out/'report.json').read_text())
    profile = json.loads((out/'target-profile.json').read_text())
    calibration = json.loads((ROOT/'hardware/performance/rocket-rv32.json').read_text())
    if (report['status'] != 'passed' or report['area_budget'] != args.area_budget or
            report['input_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or
            profile['hardware']['_profile']['clock_ns'] != args.clock or
            profile['software'] != calibration['software'] or
            profile['interface'] != {**calibration['interface'], 'max_extra_csrs': args.max_extra_csrs}):
        raise RuntimeError('Pipeline input, configuration or result does not match')
    if report.get('cosimulation') != 'passed' and report.get('software_simulation') != 'passed':
        raise RuntimeError('Selected application has not passed numerical validation')
    raw = out/'baseline.raw.mlir'
    raw.write_text((out/'input/observed-baseline.mlir').read_text())
    run('normalize-baseline', [ROOT/'build/tools/scaleisa-opt/scaleisa-opt', raw,
        '--allow-unregistered-dialect', '--scaleisa-egraph-preprocess', '-o', out/'baseline.mlir'])
    empty = out/'baseline-hardware.mlir'
    empty.write_text('module {}\n')
    run('compile-baseline', [DRIVER, out/'baseline.mlir', '--compile-only',
        f'--hardware={empty}', '--allow-no-ci', *link, '-o', out/'baseline.elf'])
    compiled = json.loads((out/'baseline.elf.scaleisa/compile-report.json').read_text())
    if compiled['ci_sites']:
        raise RuntimeError('Baseline contains CIs')
    run('baseline-simulation', [prefix/'riscv-tools/bin/spike',
        '--isa=RV32IMAFDC_Zicsr_Zifencei', out/'baseline.elf'])
    selection = report['selection']
    rejections = Counter()
    for path in (Path(report['discovery_directory'])/'uarch').glob('*/target-rejections.json'):
        rejections.update(row['reason'] for row in json.loads(path.read_text()))
    summary = {
        'status': 'passed', 'scope': f'Dimension-scaled repository {args.case} snapshot',
        'dimensions': json.loads((out/'input/provenance.json').read_text())['dimensions'],
        'max_extra_csrs': args.max_extra_csrs, 'independently_checked_values_per_binary': values,
        'selected_correctness': 'passed', 'normalized_software_baseline': 'passed; zero CIs',
        'area_budget': args.area_budget, 'scheduling_period_ns': args.clock,
        'selected_area_proxy': selection['selected_area'],
        'software_model_cycles': selection['software_only_latency'],
        'selected_model_cycles': selection['selected_latency'],
        'estimated_program_speedup': selection['software_only_latency']/selection['selected_latency'],
        'ci_sites': report['compiled']['ci_sites'], 'selected_implementations': selection['selected_impls'],
        'target_rejections': dict(rejections), 'ci_cycle_comparison': report.get('ci_cycle_comparison', []),
        'backend_pipeline_outcomes': report.get('backend_pipeline_outcomes'),
        'backend_pipeline_fallback_count': report.get('backend_pipeline_fallback_count'),
        'validation_harness': str(harness), 'chipyard_application_run': False,
        'pipeline_report': str(out/'report.json'), 'input_sha256': report['input_sha256'],
        'target_profile_sha256': report['target_profile_sha256']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
