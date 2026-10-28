#!/usr/bin/env python3
"""Validate dimension-scaled PCL ISS keypoint detection through the APS flow."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from run_octree_case import ROOT, DRIVER, observed_baseline


def prepare_input(directory: Path, size: int) -> tuple[Path, Path, tuple[str, ...]]:
    source = ROOT/'scaleisa/benchmarks/pcl/iss_keypoint'
    directory.mkdir(parents=True, exist_ok=True)
    original_cpp = (source/'iss_keypoint.cpp').read_text()
    reference, count = re.subn(r'(static const int (?:WIDTH|HEIGHT) = )40;',
                              rf'\g<1>{size};', original_cpp)
    if count != 2:
        raise RuntimeError('ISS source dimensions changed')
    (directory/'iss_reference.cpp').write_text(reference)
    original_mlir = (source/'iss_keypoint.mlir').read_text()
    mlir = original_mlir
    for old, new in [('40x', f'{size}x'), ('to 40 {', f'to {size} {{'),
                     ('constant 39 : i32', f'constant {size-1} : i32'),
                     ('PA40_', f'PA{size}_')]:
        mlir = mlir.replace(old, new)
    input_file = directory/'iss.mlir'
    input_file.write_text(mlir)
    harness = directory/'main.cpp'
    harness.write_text((ROOT/'tests/aps-scaleisa/fixtures/iss_main.cpp').read_text())
    provenance = {
        'case': 'iss_keypoint', 'original_dimensions': [40, 40],
        'dimensions': [size, size],
        'transformations': ['Spatial dimensions, loop bounds, boundary clamps and matching symbol names only; 5x5 covariance and 3x3 suppression unchanged'],
        'original_source_sha256': hashlib.sha256(original_cpp.encode()).hexdigest(),
        'original_mlir_sha256': hashlib.sha256(original_mlir.encode()).hexdigest(),
        'input_sha256': hashlib.sha256(mlir.encode()).hexdigest(),
        'scaled_source_sha256': hashlib.sha256(reference.encode()).hexdigest()}
    (directory/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    types = tuple(f'memref<{size}x{size}x{element}>' for element in ('3xf32', 'f32', 'i32'))
    return input_file, harness, types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--area-budget', type=int, default=5000)
    parser.add_argument('--clock', type=float, default=8)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'build/aps-iss-16')
    parser.add_argument('--reuse-pipeline', action='store_true')
    args = parser.parse_args()
    if args.size < 5:
        parser.error('size must accommodate the complete five-point stencil')
    out = args.output_dir.resolve()
    source, harness, types = prepare_input(out/'input', args.size)
    env = os.environ.copy()
    prefix = ROOT/'.pixi/envs/default'
    env['PATH'] = f'{prefix}/bin:{prefix}/riscv-tools/bin:{env.get("PATH", "")}'
    env['LD_LIBRARY_PATH'] = str(prefix/'lib')

    def run(stage: str, command: list) -> None:
        print(stage, flush=True)
        with (out/(stage+'.log')).open('w') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)

    run('parse-input', [ROOT/'circt/build/bin/mlir-opt', source, '-o', os.devnull])
    link = ['--emit=elf', '--entry=_start', '--abi=ilp32f',
            '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--link-source={harness}',
            f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
            f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}']
    if not args.reuse_pipeline:
        run('pipeline', [DRIVER, source, f'--area-budget={args.area_budget}',
            f'--output-dir={out}', f'--clock={args.clock}', '--cosim',
            '--rename-main=pcl_selected_main', '--observe-function=pcl_validate_outputs',
            *[f'--observe-type={typ}' for typ in types], *link])
    report = json.loads((out/'report.json').read_text())
    if report['status'] != 'passed':
        raise RuntimeError('Pipeline did not pass')
    if report['input_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError('Pipeline input does not match')
    if report['area_budget'] != args.area_budget:
        raise RuntimeError('Pipeline budget does not match')
    profile = json.loads((out/'target-profile.json').read_text())
    calibration = json.loads((ROOT/'hardware/performance/rocket-rv32.json').read_text())
    if profile['hardware']['_profile']['clock_ns'] != args.clock or any(
            profile[key] != calibration[key] for key in ('software', 'interface')):
        raise RuntimeError('Pipeline profile does not match current settings')
    if report.get('cosimulation') != 'passed' and report.get('software_simulation') != 'passed':
        raise RuntimeError('Selected program has not passed numerical validation')
    raw = out/'baseline.raw.mlir'
    raw.write_text(observed_baseline(source, types))
    run('normalize-baseline', [ROOT/'build/tools/scaleisa-opt/scaleisa-opt', raw,
        '--scaleisa-egraph-preprocess', '-o', out/'baseline.mlir'])
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
    target_rejections = Counter()
    for path in (Path(report['discovery_directory'])/'uarch').glob('*/target-rejections.json'):
        target_rejections.update(row['reason'] for row in json.loads(path.read_text()))
    summary = {
        'status': 'passed', 'scope': 'Dimension-scaled repository ISS keypoint snapshot',
        'dimensions': [args.size, args.size], 'covariance_stencil': [5, 5],
        'suppression_stencil': [3, 3], 'independently_checked_values_per_binary': args.size**2*5,
        'selected_correctness': 'passed', 'normalized_software_baseline': 'passed; zero CIs',
        'area_budget': args.area_budget, 'scheduling_period_ns': args.clock,
        'selected_area_proxy': selection['selected_area'],
        'software_model_cycles': selection['software_only_latency'],
        'selected_model_cycles': selection['selected_latency'],
        'estimated_program_speedup': selection['software_only_latency']/selection['selected_latency'],
        'ci_sites': report['compiled']['ci_sites'], 'selected_implementations': selection['selected_impls'],
        'target_rejections': dict(target_rejections),
        'ci_cycle_comparison': report.get('ci_cycle_comparison', []),
        'validation_harness': str(harness), 'chipyard_application_run': False,
        'pipeline_report': str(out/'report.json'), 'input_sha256': report['input_sha256'],
        'target_profile_sha256': report['target_profile_sha256']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
