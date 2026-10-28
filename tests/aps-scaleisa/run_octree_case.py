#!/usr/bin/env python3
"""Run full source-derived octree discovery, RTL validation and a CPU baseline."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scaleisa/python'))
from scaleisa_egraph.mlir.module_text import _extract_function_blocks

SOURCE = ROOT / 'scaleisa/benchmarks/pcl/octree_search/octree_search.mlir'
HARNESS = ROOT / 'tests/aps-scaleisa/fixtures/octree_main.cpp'
DRIVER = ROOT / 'build/tools/aps-scaleisa-opt/aps-scaleisa-opt'
TYPES = ('memref<2048x3xf32>', 'memref<4x4096xi32>', 'memref<128xi32>')


def observed_baseline(source: Path = SOURCE, types: tuple[str, ...] = TYPES) -> str:
    """Attach the same output observer used by the driver to the original main."""
    text = source.read_text()
    main = _extract_function_blocks(text)['main']
    values = []
    for typ in types:
        matches = re.findall(r'(%[\w.$-]+) = memref\.alloca\([^\n]* : ' +
                             re.escape(typ) + r'\s*$', main, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f'Expected one entry allocation of {typ}')
        values.extend(matches)
    observed = main.replace('@main(', '@pcl_selected_main(', 1)
    call = '    func.call @pcl_validate_outputs(' + ', '.join(values) + ') : (' + ', '.join(types) + ') -> ()\n'
    observed, count = re.subn(r'(?m)^(\s*return [^\n]+)$', lambda m: call + m[0], observed)
    if count != 1:
        raise RuntimeError('Expected one main return for the validation observer')
    text = text.replace(main, observed, 1)
    declaration = '  func.func private @pcl_validate_outputs(' + ', '.join(types) + ')\n'
    head, tail = text.rsplit('}', 1)
    return head + declaration + '}' + tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--area-budget', type=int, default=5000)
    parser.add_argument('--clock', type=float, help='Scheduling period; defaults to the ScaleISA profile')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'build/aps-octree-asap7')
    parser.add_argument('--reuse-pipeline', action='store_true')
    args = parser.parse_args()
    physical = json.loads((ROOT/'scaleisa/resources/scaleisa-hector/resource.json').read_text())
    if args.clock is None:
        args.clock = physical['_profile']['clock_ns']
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prefix = ROOT/'.pixi/envs/default'
    env['PATH'] = os.pathsep.join([str(prefix/'bin'), str(prefix/'riscv-tools/bin'), env.get('PATH', '')])
    env['LD_LIBRARY_PATH'] = str(prefix/'lib')

    def run(name, command):
        print(name, flush=True)
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)

    link = ['--emit=elf', '--entry=_start', '--abi=ilp32f',
            '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--link-source={HARNESS}',
            f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
            f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}']
    if not args.reuse_pipeline:
        run('pipeline', [DRIVER, SOURCE, f'--area-budget={args.area_budget}',
            f'--output-dir={out}', f'--clock={args.clock}', '--cosim', '--rename-main=pcl_selected_main',
            '--observe-function=pcl_validate_outputs',
            *[f'--observe-type={typ}' for typ in TYPES], *link])
    report = json.loads((out/'report.json').read_text())
    if report['status'] != 'passed':
        raise RuntimeError('The pipeline has not passed')
    if report['input_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
        raise RuntimeError('Pipeline input does not match the benchmark')
    if report['area_budget'] != args.area_budget:
        raise RuntimeError('Pipeline area budget does not match')
    current_profile = json.loads((ROOT/'hardware/performance/rocket-rv32.json').read_text())
    realized_profile = json.loads((out/'target-profile.json').read_text())
    if realized_profile['hardware']['_profile']['clock_ns'] != args.clock:
        raise RuntimeError('Pipeline scheduling period does not match')
    for operation in ('addi', 'muli', 'divsi', 'remsi'):
        if realized_profile['hardware'][operation]['delay'] != physical[operation]['delay']:
            raise RuntimeError('Pipeline does not use the current ASAP7 delay table')
    if any(realized_profile[key] != current_profile[key] for key in ('software', 'interface')):
        raise RuntimeError('Pipeline target profile does not match current calibration')
    if report.get('cosimulation') != 'passed' and report.get('software_simulation') != 'passed':
        raise RuntimeError('Selected program has not passed validation')

    raw = out/'baseline.raw.mlir'
    raw.write_text(observed_baseline())
    run('normalize-baseline', [ROOT/'build/tools/scaleisa-opt/scaleisa-opt', raw,
        '--scaleisa-egraph-preprocess', '-o', out/'baseline.mlir'])
    empty_hardware = out/'baseline-hardware.mlir'
    empty_hardware.write_text('module {}\n')
    run('compile-baseline', [DRIVER, out/'baseline.mlir', '--compile-only',
        f'--hardware={empty_hardware}', '--allow-no-ci', *link, '-o', out/'baseline.elf'])
    compiled = json.loads((out/'baseline.elf.scaleisa/compile-report.json').read_text())
    if compiled['ci_sites'] != 0:
        raise RuntimeError('Software baseline contains custom instructions')
    run('baseline-simulation', [prefix/'riscv-tools/bin/spike', '--isa=RV32IMAFDC_Zicsr_Zifencei', out/'baseline.elf'])
    selection = report['selection']
    rejections = Counter()
    for implementation_report in (Path(report['discovery_directory'])/'uarch').glob('*/implementation-report.json'):
        for row in json.loads(implementation_report.read_text())['implementations']:
            if row['status'] != 'ok':
                rejections[row.get('reason', row['status'])] += 1
    summary = {'status': 'passed', 'scope': 'Full octree_search snapshot; hierarchical occupancy and neighborhood queries',
        'points': 2048, 'levels': 4, 'grid': 16, 'queries': 128,
        'independently_checked_values_per_binary': 22656, 'expected_signature': 17565906,
        'selected_correctness': 'passed', 'normalized_software_baseline': 'passed; zero CIs',
        'area_budget': args.area_budget, 'scheduling_period_ns': args.clock,
        'selected_area_proxy': selection['selected_area'],
        'software_model_cycles': selection['software_only_latency'],
        'selected_model_cycles': selection['selected_latency'],
        'estimated_program_speedup': selection['software_only_latency']/selection['selected_latency'],
        'speedup_scope': 'ScaleISA program estimate; RTL CI cycles measured separately, CPU cycles excluded',
        'ci_sites': report['compiled']['ci_sites'], 'selected_implementations': selection['selected_impls'],
        'ci_cycle_comparison': report.get('ci_cycle_comparison', []),
        'implementation_rejections': dict(rejections),
        'chipyard_application_run': False, 'pipeline_report': str(out/'report.json'),
        'input_sha256': report['input_sha256'], 'target_profile_sha256': report['target_profile_sha256']}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
