#!/usr/bin/env python3
"""Validate small source-derived scalar kernels and report modeled speedups.

The test scaffolding supplies deterministic inputs and independent C++ oracles.
aps-scaleisa-opt performs all AU discovery, insertion, hardware and CI generation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scaleisa/python'))
from scaleisa_egraph.mlir.module_text import _extract_function_blocks

DRIVER = ROOT / 'build/tools/aps-scaleisa-opt/aps-scaleisa-opt'
ORIGINS = {
    'clamp': ('pcl/iss_keypoint/iss_keypoint.mlir', '_ZL7clamp_iiii'),
    'bitcount': ('edgebench/polygeist/orb_descriptor_match/orb_descriptor_match.mlir', 'bitcount'),
    'sq_dist': ('pcl/kdtree_search/kdtree_search.mlir', '_ZL7sq_distPfS_'),
}


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def values_for(name, count):
    if name == 'clamp':
        return [[(i * 37) % 257 - 128, i % 7 - 9, i % 7 - 8 + i % 23]
                for i in range(count)]
    if name == 'bitcount':
        edges = [0, 1, 0xffffffff, 0x80000000, 0x7fffffff, 0x55555555, 0xaaaaaaaa, 0x01010101]
        return [[edges[i] if i < len(edges) else
                 ((i * 0x9e3779b9) ^ ((i + 17) << 19)) & 0xffffffff]
                for i in range(count)]
    return [[f32(((i * 37 + c * 11) % 101 - 50) * f32(.031)) for c in range(3)] +
            [f32(((i * 19 + c * 17) % 97 - 48) * f32(.029)) for c in range(3)]
            for i in range(count)]


def prepare(name, count, directory):
    origin, symbol = ORIGINS[name]
    source = ROOT / 'scaleisa/benchmarks' / origin
    kernel = _extract_function_blocks(source.read_text())[symbol]
    rows = values_for(name, count)
    floating = name == 'sq_dist'
    elem = 'f32' if floating else 'i32'
    cols = len(rows[0])
    shape = f'{count}x{cols}x{elem}'
    result_type = f'memref<{count}x{elem}>'

    def literal(value):
        if floating:
            return f'{value:.9e}'
        return str(value if value < 2**31 else value - 2**32)

    dense = '[' + ','.join('[' + ','.join(map(literal, row)) + ']' for row in rows) + ']'
    output_global = f'memref.global @scalar_results : {result_type} = uninitialized'
    global_op = f'memref.global "private" constant @inputs : memref<{shape}> = dense<{dense}>'
    prelude = [f'%data = memref.get_global @inputs : memref<{shape}>',
               f'%out = memref.get_global @scalar_results : {result_type}', '%zero = arith.constant 0 : i32']
    operations = [f'%v{c} = affine.load %data[%i,{c}] : memref<{shape}>' for c in range(cols)]
    if name == 'sq_dist':
        prelude += ['%a = memref.alloca() : memref<3xf32>',
                    '%b = memref.alloca() : memref<3xf32>',
                    '%ad = memref.cast %a : memref<3xf32> to memref<?xf32>',
                    '%bd = memref.cast %b : memref<3xf32> to memref<?xf32>']
        operations += [f'affine.store %v{c}, %a[{c}] : memref<3xf32>' for c in range(3)]
        operations += [f'affine.store %v{c+3}, %b[{c}] : memref<3xf32>' for c in range(3)]
        call = f'%r = func.call @{symbol}(%ad,%bd) : (memref<?xf32>,memref<?xf32>)->f32'
    else:
        call = f'%r = func.call @{symbol}(' + ','.join(f'%v{c}' for c in range(cols)) + ') : (' + ','.join(['i32']*cols) + ')->i32'
    operations += [call, f'affine.store %r,%out[%i] : {result_type}']
    main = 'func.func @main()->i32 {\n' + '\n'.join(prelude) + f'\naffine.for %i=0 to {count} {{\n' + '\n'.join(operations) + '\n}\nreturn %zero : i32\n}'
    module = 'module {\n' + global_op + '\n' + output_global + '\n' + kernel + '\n' + main + '\n}\n'
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / 'input.mlir'
    affine_input = directory / 'input.affine.mlir'
    affine_input.write_text(module)
    subprocess.run([str(ROOT/'circt/build/bin/mlir-opt'), str(affine_input),
                    '--lower-affine', '-o', str(input_path)], check=True)

    baseline_main = main.replace('@main()', '@scalar_selected_main()', 1)
    (directory/'baseline.raw.mlir').write_text('module {\n'+global_op+'\n'+output_global+'\n'+kernel+'\n'+baseline_main+'\n}\n')
    crows = '{' + ','.join('{' + ','.join(f'{v:.9e}f' if floating else f'{v}u' for v in row) + '}' for row in rows) + '}'
    if name == 'clamp':
        crows = '{' + ','.join('{' + ','.join(map(str, row)) + '}' for row in rows) + '}'
        oracle = 'int32_t x=inputs[i][0],lo=inputs[i][1],hi=inputs[i][2]; return x<lo?lo:x>hi?hi:x;'
        ctype = 'int32_t'
    elif name == 'bitcount':
        oracle = 'uint32_t x=inputs[i][0]; int32_t n=0; while(x){x&=x-1;++n;} return n;'
        ctype = 'uint32_t'
    else:
        oracle = 'float x=inputs[i][0]-inputs[i][3],y=inputs[i][1]-inputs[i][4],z=inputs[i][2]-inputs[i][5]; return (x*x+y*y)+z*z;'
        ctype = 'float'
    result_c = 'float' if floating else 'int32_t'
    compare = 'bits(values[i])!=bits(reference(i))' if floating else 'values[i]!=reference(i)'
    harness = f'''#include <stdint.h>
static const {ctype} inputs[{count}][{cols}]={crows};
volatile int test_status,checked_outputs,validation_calls,mismatches;
volatile int first_mismatch=-1;
static uint32_t bits(float x){{uint32_t b;__builtin_memcpy(&b,&x,4);return b;}}
static {result_c} reference(int i){{{oracle}}}
extern "C" int scalar_selected_main();
extern "C" {result_c} scalar_results[{count}];
static void scalar_validate(){{
 {result_c} *values=scalar_results;
 ++validation_calls;
 for(int i=0;i<{count};++i){{
  if({compare}){{if(first_mismatch<0)first_mismatch=i;++mismatches;}}
  ++checked_outputs;
 }}
}}
extern "C" int main(){{
 int result=scalar_selected_main();
 scalar_validate();
 return result || validation_calls!=1 || checked_outputs!={count} || mismatches;
}}
'''
    (directory/'main.cpp').write_text(harness)
    (directory/'provenance.json').write_text(json.dumps({
        'scope': 'kernel microbenchmark, not the complete source application',
        'origin': str(source.relative_to(ROOT)), 'symbol': symbol,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'kernel_sha256': hashlib.sha256(kernel.encode()).hexdigest(),
        'samples': count, 'generation': 'Original kernel computation retained; deterministic caller and independent C++ oracle added',
        'normalization': 'Standard lower-affine pass before SCF-only discovery',
        'observability': 'Public result global exists before discovery and is checked directly by C++ caller',
    }, indent=2)+'\n')
    return input_path, result_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=list(ORIGINS),
                        default=['bitcount', 'sq_dist'])
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--area-budget', type=int, default=500)
    parser.add_argument('--discovery-stages', default='linalg,affine,scf')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'build/aps-scaleisa-scalars')
    args = parser.parse_args()
    if not 8 <= args.samples <= 256:
        parser.error('--samples must be between 8 and 256')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = ROOT/'scaleisa/tmp'/('scalar-inputs-'+output.name)
    summary = {'status': 'running', 'scope': 'scalar kernel microbenchmarks',
        'area_budget': args.area_budget, 'samples_per_case': args.samples,
        'baseline_preprocessing': 'shared ScaleISA preprocessing; zero custom instructions',
        'speedup_method': 'software_only_latency / selected_latency from the same ScaleISA serial latency model; not measured processor speedup or simulator wall-time speedup',
        'cases': []}
    summary_path = output/'summary.json'
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join([str(ROOT/'.pixi/envs/default/bin'),
                                  str(ROOT/'.pixi/envs/default/riscv-tools/bin'),
                                  env.get('PATH', '')])
    env['LD_LIBRARY_PATH'] = str(ROOT/'.pixi/envs/default/lib')
    def run(label, command, directory):
        print(label, flush=True)
        with (directory/(label+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if result.returncode:
            raise RuntimeError(f'{label}: exit {result.returncode}; see {directory/(label+".log")}')
    for name in args.cases:
        case = {'name': name, 'status': 'running'}
        summary['cases'].append(case)
        directory = output/name
        directory.mkdir(exist_ok=True)
        try:
            source, result_type = prepare(name, args.samples, inputs/name)
            case['input'] = str(source)
            run('pipeline', [DRIVER, source, f'--area-budget={args.area_budget}',
                f'--discovery-stages={args.discovery_stages}', f'--output-dir={directory}', '--emit=elf',
                '--entry=_start', '--abi=ilp32f', '--mattr=+m,+a,+f,+zicsr,+zifencei',
                '--cosim', '--rename-main=scalar_selected_main', f'--link-source={source.parent/"main.cpp"}',
                f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
                f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}'], directory)
            report = json.loads((directory/'report.json').read_text())
            assert report['status']=='passed'
            assert report['cosimulation']=='passed' or report.get('software_simulation')=='passed'
            run('normalize-baseline', [ROOT/'build/tools/scaleisa-opt/scaleisa-opt',
                source.parent/'baseline.raw.mlir', '--scaleisa-egraph-preprocess',
                '-o', directory/'baseline.mlir'], directory)
            run('compile-baseline', [DRIVER, directory/'baseline.mlir', '--compile-only',
                f'--hardware={directory/"hardware.mlir"}', '--allow-no-ci', '--emit=elf', '--entry=_start',
                '--abi=ilp32f', '--mattr=+m,+a,+f,+zicsr,+zifencei',
                f'--link-source={source.parent/"main.cpp"}', f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
                f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}', '-o', directory/'baseline.elf'], directory)
            baseline = json.loads((directory/'baseline.elf.scaleisa/compile-report.json').read_text())
            assert baseline['ci_sites']==0
            simulator = ([directory/'simulator/Vmain'] if report.get('hardware_selected', True)
                         else [ROOT/'.pixi/envs/default/riscv-tools/bin/spike',
                               '--isa=RV32IMAFDC_Zicsr_Zifencei'])
            run('baseline-simulation', [*simulator, directory/'baseline.elf'], directory)
            s = report['selection']
            case.update(status='passed', checks=args.samples,
                selected_kernels=len(s['selected_impls']), selected_area=s['selected_area'],
                software_model_cycles=s['software_only_latency'], selected_model_cycles=s['selected_latency'],
                estimated_cycle_speedup=s['software_only_latency']/s['selected_latency'],
                simulation_seconds=next(c['seconds'] for c in report['commands'] if c['stage'] in ('simulation', 'software-simulation')),
                ci_sites=report['compiled']['ci_sites'], discovery_metrics=report['discovery_metrics'],
                ci_cycle_comparison=report.get('ci_cycle_comparison', []),
                baseline_ci_sites=0, baseline_validation='passed', pipeline_report=str(directory/'report.json'))
        except Exception as error:
            case.update(status='failed', error=str(error))
            print(str(error), file=sys.stderr, flush=True)
        summary_path.write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(case), flush=True)
    summary['status'] = 'passed' if all(c['status']=='passed' for c in summary['cases']) else 'failed'
    summary_path.write_text(json.dumps(summary, indent=2)+'\n')
    print(summary_path)
    return 0 if summary['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
