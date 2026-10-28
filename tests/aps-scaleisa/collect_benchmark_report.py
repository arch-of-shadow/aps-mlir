#!/usr/bin/env python3
"""Archive exact measured compiler artifacts and build a linked speedup table."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(entry: dict, output: Path) -> dict:
    source = ROOT / entry['source']
    measured_path = ROOT / entry['chipyard']
    measured = json.loads(measured_path.read_text())
    pipeline = json.loads((source / 'report.json').read_text())
    if measured['status'] != 'passed' or pipeline['status'] != 'passed':
        raise ValueError('Native and Chipyard runs must both pass')
    if pipeline.get('cosimulation') != 'passed' and pipeline.get('software_simulation') != 'passed':
        raise ValueError('Missing numerical validation')
    if digest(source / 'design.sv') != measured['rtl_sha256']:
        raise ValueError('Measured RTL differs from selected native RTL')
    for name in ('candidates.mlir', 'selected.mlir', 'hardware.mlir', 'design.sv'):
        original = source / name
        expected = pipeline.get('hashes', {}).get(str(original))
        if expected and digest(original) != expected:
            raise ValueError(f'Pipeline artifact changed: {original}')
    for mode, expected in measured['binary_sha256'].items():
        if digest(measured_path.parent / (mode + '.riscv')) != expected:
            raise ValueError(f'Measured {mode} binary changed')
    directory = output / entry['id']
    directory.mkdir(parents=True, exist_ok=True)
    files = []

    def copy(original: Path, destination: str) -> None:
        target = directory / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        files.append(dict(path=destination, source=str(original.relative_to(ROOT)), sha256=digest(target)))

    for name, destination in [('candidates.mlir', 'scaleisa-au.mlir'),
                              ('planned.mlir', 'aps-planned.mlir'),
                              ('hardware.mlir', 'aps-hardware.mlir'),
                              ('selected.mlir', 'selected-program.mlir'),
                              ('baseline.mlir', 'baseline-program.mlir'),
                              ('target-profile.json', 'target-profile.json'),
                              ('report.json', 'pipeline-report.json')]:
        copy(source / name, destination)
    original_input = Path(pipeline['input'])
    if digest(original_input) != pipeline['input_sha256']:
        raise ValueError('Original benchmark input changed')
    copy(original_input, 'input.mlir')
    for name in ('provenance.json', 'main.cpp', 'reference.cpp', 'iss_reference.cpp'):
        if (original_input.parent / name).exists():
            copy(original_input.parent / name, 'input/' + name)
    copy(measured_path, 'chipyard-result.json')
    if measured.get('validation_harness'):
        copy(Path(measured['validation_harness']), 'validation-harness.cpp')
    for name in ('design.scheduled-tor.mlir', 'ci-runtime.json'):
        if (source / name).exists():
            copy(source / name, name)
    for mode in ('baseline', 'accelerated'):
        copy(measured_path.parent / 'output' / (mode + '.log'), mode + '-chipyard.log')
    ci_path = measured_path.parent / 'accelerated.riscv.scaleisa/ci.mlir'
    copy(ci_path, 'cpu-ci.mlir')
    implementations = pipeline['selection']['selected_impls']
    cache = Path(pipeline['discovery_directory'])
    index = ['# ' + entry['name'], '',
             'All MLIR files are byte-for-byte copies of the measured run artifacts. '
             'Hashes and source paths are in [manifest.json](manifest.json).', '',
             '[Collected AU definitions](scaleisa-au.mlir) · '
             '[APS hardware](aps-hardware.mlir) · [APS planning](aps-planned.mlir) · '
             '[CPU CI lowering](cpu-ci.mlir) · [Selected program](selected-program.mlir)', '',
             '| Selected AU / APS function | ScaleISA implementation input | Hector evaluated schedule |',
             '|---|---|---|']
    for impl in implementations:
        name, stage = impl['function_name'], impl['origin_stage']
        links = []
        for suffix in ('input', 'evaluated'):
            original = cache / 'uarch' / stage / (name + '.' + suffix + '.mlir')
            relative = 'au/' + original.name
            copy(original, relative)
            links.append(f'[{suffix}]({relative})')
        index.append(f'| `{name}` | {links[0]} | {links[1]} |')
    if not implementations:
        index.extend(['', '**Software-only selection: no AU or CI was selected.**'])
    if (source / 'design.scheduled-tor.mlir').exists():
        index.extend(['', '[Scheduled APS/TOR and realized memory map](design.scheduled-tor.mlir)'])
    cycles = measured['cycles']
    selection = pipeline['selection']
    model = selection['software_only_latency'] / selection['selected_latency']
    actual = cycles['baseline']['warm'] / cycles['accelerated']['warm']
    if abs(model - measured['scaleisa_estimated_speedup']) > 1e-9:
        raise ValueError('Measured run and selection refer to different model estimates')
    row = dict(id=entry['id'], name=entry['name'], status='passed',
               selected_au_count=len(implementations), static_ci_sites=pipeline['compiled']['ci_sites'],
               chipyard_cycles=cycles, measured_speedup=actual, estimated_speedup=model,
               model_cycles=dict(baseline=selection['software_only_latency'], accelerated=selection['selected_latency']),
               area_budget=pipeline['area_budget'], selected_area=selection['selected_area'],
               clock_ns=json.loads((source / 'target-profile.json').read_text())['hardware']['_profile']['clock_ns'],
               backend_pipeline_fallback_count=pipeline.get('backend_pipeline_fallback_count'),
               correctness=measured.get('correctness',
                   f"{measured.get('validated_outputs_per_binary')} independently checked scalar outputs per binary"),
               source=entry['source'], chipyard=entry['chipyard'], files=files)
    (directory / 'manifest.json').write_text(json.dumps(row, indent=2) + '\n')
    (directory / 'index.md').write_text('\n'.join(index) + '\n')
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('benchmark_suite16.json'))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'build/aps-suite-16/review')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())
    if len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('Duplicate test case')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for entry in entries:
        if args.allow_incomplete and not (ROOT / entry['chipyard']).exists():
            missing.append(entry['id'])
            continue
        rows.append(collect(entry, output))
    report = ['# ScaleISA → APS: measured benchmark comparison', '',
              'Measured speedup = warm software CPU cycles / warm accelerated CPU cycles '
              'on full Chipyard APSRocketConfig Verilator (`run-binary-debug LOADMEM=1`). '
              'ScaleISA speedup = its modeled software cycles / modeled selected-program cycles. '
              'Reference generation, validation and printing are outside the timed regions. '
              'These are cycle speedups; scheduling period is recorded separately and is not a post-layout frequency result.', '',
              'Each case index lists every selected AU, with its exact implementation input and '
              'Hector schedule. The APS link contains all corresponding hardware functions. '
              'CPU CI lowering and file hashes are included in each case directory.', '',
              '| Case | AUs / CI sites | HLS period (ns) | CPU cycles baseline → accelerated | Measured speedup | ScaleISA estimate | ScaleISA AUs | APS dialect |',
              '|---|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        name = row['id']
        report.append(f"| {row['name']} | {row['selected_au_count']} / {row['static_ci_sites']} | "
                      f"{row['clock_ns']:g} | "
                      f"{row['chipyard_cycles']['baseline']['warm']:,} → {row['chipyard_cycles']['accelerated']['warm']:,} | "
                      f"{row['measured_speedup']:.3f}× | {row['estimated_speedup']:.3f}× | "
                      f"[AU MLIR]({name}/scaleisa-au.mlir), [index]({name}/index.md) | [APS]({name}/aps-hardware.mlir) |")
    report.extend(['', 'Pose2 and Pose3 rows time their solvers and bounded input staging; '
                   'graph construction runs outside the timed region. Bitcount and squared '
                   'distance are kernel microbenchmarks. Other rows time the complete '
                   'repository snapshot entry point. All selected CI implementations are '
                   'included in the linked AU and APS modules.', '',
                   'Area budgets, selected area estimates, cold measurements and exact '
                   'profiles are retained in [summary.json](summary.json).'])
    if (output / 'result-comparisons.json').exists():
        report.extend(['', '[Executed numerical software-reference comparisons](result-comparisons.json).'])
    if (output / 'correctness.md').exists():
        report.extend(['', '[Numerical software-reference comparisons and wrong-result controls](correctness.md).'])
    if missing:
        report.extend(['', '**Incomplete: pending cases:** ' + ', '.join(missing)])
    result = dict(status='incomplete' if missing else 'passed', case_count=len(rows),
                  distinct_case_count=len({r['id'] for r in rows}), missing=missing, cases=rows)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    print(output / 'report.md')


if __name__ == '__main__':
    main()
