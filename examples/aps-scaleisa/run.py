#!/usr/bin/env python3
"""Run source-to-RTL APS/ScaleISA evaluations with independent output validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

# Keep Python's generated cache outside the source examples.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRIVER = ROOT / 'build/tools/aps-scaleisa-opt/aps-scaleisa-opt'
CHIPYARD = ROOT / 'thirdparty/chipyard'
FIXTURES = HERE / 'fixtures'


def environment(jobs=None):
    prefix = ROOT / '.pixi/envs/default'
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join([str(prefix / 'bin'), str(prefix / 'riscv-tools/bin'),
                                   str(ROOT / 'thirdparty/polygeist'), env.get('PATH', '')])
    env['LD_LIBRARY_PATH'] = os.pathsep.join([str(prefix / 'lib'),
        str(prefix / 'riscv-tools/lib'), env.get('LD_LIBRARY_PATH', '')])
    env.update(RISCV=str(prefix / 'riscv-tools'),
        LEGACY_FIRTOOL=str(ROOT / 'thirdparty/firtool-1.75.0/bin/firtool'),
        JAVA_TOOL_OPTIONS=f'-Xmx8G -Xss8M -Djava.io.tmpdir={CHIPYARD}/.java_tmp -XX:ActiveProcessorCount={jobs or os.cpu_count() or 1}')
    return env


class Commands:
    """Record commands, logs and failures consistently for every evaluation stage."""
    def __init__(self, directory, jobs=None):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.env = environment(jobs)
        self.records = []
        self.lock = threading.Lock()

    def run(self, stage, command):
        command = list(map(str, command))
        print(stage, flush=True)
        started = time.monotonic()
        log_path = self.directory / (stage + '.log')
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=self.env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=14400)
        with self.lock:
            self.records.append(dict(stage=stage, argv=command, returncode=result.returncode,
                                     seconds=time.monotonic() - started))
            (self.directory / 'commands.json').write_text(json.dumps(self.records, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{stage} failed with exit {result.returncode}; see {log_path}')


def link_arguments(max_extra_csrs, harness):
    return ['--emit=elf', '--entry=_start', '--abi=ilp32f',
            '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--max-extra-csrs={max_extra_csrs}',
            f'--link-source={harness}', f'--link-source={FIXTURES / "start.S"}',
            f'--link-arg=-T{FIXTURES / "ram.ld"}']


def prepare_case(case, output, jobs):
    """Generate discovery input and an observed software baseline from C++."""
    executor = Commands(output / 'prepare', jobs)
    raw, clean = output / 'raw.mlir', output / 'clean.mlir'
    lifted, source = output / 'lifted.mlir', output / 'input.mlir'
    cgeist = shutil.which('cgeist', path=executor.env['PATH'])
    if not cgeist:
        raise RuntimeError('cgeist is required; run the repository setup first')
    executor.run('cgeist', [cgeist, '-S', '-O0', '--immediate', '--raise-scf-to-affine',
        '-x', 'c++', '-std=c++17', '-ffp-contract=off', case['directory'] / 'source.cpp', '-o', raw])
    # Discard the host target layout before importing into the RV32 toolchain.
    raw.write_text(re.sub(r'\Amodule attributes \{[^\n]*\} \{', 'module {', raw.read_text(), count=1))
    cleanup = ROOT / 'scaleisa/benchmarks/pcl/scripts/cleanup_feature_matching_polygeist_mlir.py'
    executor.run('cleanup', [sys.executable, cleanup, raw, clean])
    mlir_opt = ROOT / 'circt/build/bin/mlir-opt'
    executor.run('lift-control-flow', [mlir_opt, clean, '--allow-unregistered-dialect',
        '--lift-cf-to-scf', '--mem2reg', '--canonicalize', '--cse', '-o', lifted])
    executor.run('cleanup-lifted', [sys.executable, cleanup, lifted, clean])
    executor.run('canonicalize', [mlir_opt, clean, '--allow-unregistered-dialect',
                                '--canonicalize', '--cse', '-o', source])
    (output / 'baseline.raw.mlir').write_text(observed_baseline(source.read_text(), case['observe']))


def observed_baseline(text, observe):
    """Mirror the driver's entry renaming and output observer on the baseline."""
    sys.path.insert(0, str(ROOT / 'scaleisa/python'))
    from scaleisa_egraph.mlir.module_text import _extract_function_blocks
    main = _extract_function_blocks(text)['main']
    observed = main.replace('@main(', '@example_main(', 1)
    if not observe:
        return text.replace(main, observed, 1)
    allocations = re.findall(r'(%[\w.$-]+) = memref\.alloca\([^\n]* : (memref<[^>]+>)', main)
    values = []
    for index in observe.get('allocations', []):
        if index >= len(allocations):
            raise ValueError('Validation allocation is missing from the compiled entry')
        values.append(allocations[index])
    for typ in observe.get('types', []):
        matches = [item for item in allocations if item[1] == typ]
        if len(matches) != 1:
            raise ValueError(f'Validation requires exactly one entry allocation of {typ}')
        values.extend(matches)
    types = ', '.join(typ for _, typ in values)
    call = '    func.call @example_validate(' + ', '.join(value for value, _ in values) + ') : (' + types + ') -> ()\n'
    observed, count = re.subn(r'(?m)^(\s*(?:func\.)?return [^\n]+)$', lambda m: call + m[0], observed)
    if count != 1:
        raise ValueError('Validation requires one entry return')
    text = text.replace(main, observed, 1)
    head, tail = text.rsplit('}', 1)
    return head + '  func.func private @example_validate(' + types + ')\n}' + tail

def load_case(name: str) -> dict:
    directory = HERE / 'cases' / name
    config = {'area_budget': 10000, 'clock': 8, 'max_extra_csrs': 5, 'validation_boundary': 'observer'}
    config.update(json.loads((directory / 'case.json').read_text()))
    for filename in ['source.cpp', 'main.cpp']:
        if not (directory / filename).is_file():
            raise ValueError(f'{name}: missing {filename}')
    return {'id': name, 'directory': directory, **config}


def native_commands(case: dict, output: Path, jobs: int | None = None) -> list[tuple[str, list[str]]]:
    directory = case['directory']
    link = link_arguments(case['max_extra_csrs'], directory / 'main.cpp')
    observe = case['observe']
    instrumentation = []
    if observe:
        instrumentation = ['--observe-function=example_validate']
        instrumentation += [f'--observe-type={ty}' for ty in observe.get('types', [])]
        instrumentation += [f'--observe-allocation={i}' for i in observe.get('allocations', [])]
    pipeline = [str(DRIVER), str(output / 'input.mlir'),
                f'--area-budget={case["area_budget"]}', f'--clock={case["clock"]}',
                f'--output-dir={output}', '--cosim', '--rename-main=example_main',
                '--timeout=7200', *instrumentation, *link]
    if jobs is not None:
        pipeline += [f'--jobs={jobs}']
    return [
        ('prepare', [sys.executable, str(HERE / 'run.py'), '--prepare-case', case['id'],
                     '--output-dir', str(output)]),
        ('normalize-baseline', [str(ROOT / 'build/tools/scaleisa-opt/scaleisa-opt'),
            str(output / 'baseline.raw.mlir'), '--allow-unregistered-dialect',
            '--scaleisa-egraph-preprocess', '-o', str(output / 'baseline.mlir')]),
        ('compile-baseline', [str(DRIVER), str(output / 'baseline.mlir'), '--compile-only',
            f'--hardware={output / "baseline-hardware.mlir"}', '--allow-no-ci',
            *link, '-o', str(output / 'baseline.elf')]),
        ('validate-baseline', [str(ROOT / '.pixi/envs/default/riscv-tools/bin/spike'),
            '--isa=RV32IMAFDC_Zicsr_Zifencei', str(output / 'baseline.elf')]),
        ('pipeline', pipeline),
    ]


def command_plan(case: dict, output: Path, chipyard: bool, jobs: int | None) -> list[tuple[str, list[str]]]:
    native = output / case['id'] / 'native'
    plan = native_commands(case, native, jobs)
    if chipyard:
        command = [sys.executable, str(HERE / 'run.py'), '--measure-rocket',
                   '--source-dir', str(native),
                   '--output-dir', str(output / case['id'] / 'chipyard')]
        if jobs is not None:
            command += ['--jobs', str(jobs)]
        plan.append(('rocket', command))
    return plan


def run_case(case: dict, output: Path, chipyard: bool, jobs: int | None) -> None:
    native = output / case['id'] / 'native'
    native.mkdir(parents=True, exist_ok=True)
    (native / 'baseline-hardware.mlir').write_text('module {}\n')
    executor = Commands(native, jobs)
    for stage, command in command_plan(case, output, chipyard, jobs):
        executor.run(stage, command)
        if stage == 'compile-baseline':
            compiled = json.loads((native / 'baseline.elf.scaleisa/compile-report.json').read_text())
            if compiled['ci_sites'] != 0:
                raise RuntimeError('Software baseline contains custom instructions')
        if stage == 'pipeline':
            report = json.loads((native / 'report.json').read_text())
            if report['status'] != 'passed' or not (
                report.get('cosimulation') == 'passed' or report.get('software_simulation') == 'passed'
            ):
                raise RuntimeError('Accelerated program has not passed independent output validation')
            selection = report['selection']
            summary = dict(status='passed', ci_sites=report['compiled']['ci_sites'],
                independently_checked_values_per_binary=case['checked_values'],
                validation_harness=str(case['directory'] / 'main.cpp'),
                validation_boundary=case['validation_boundary'],
                pipeline_report=str(native / 'report.json'),
                input_sha256=hashlib.sha256((native / 'input.mlir').read_bytes()).hexdigest(),
                target_profile_sha256=report['target_profile_sha256'],
                estimated_program_speedup=selection['software_only_latency'] / selection['selected_latency'],
                scheduling_period_ns=case['clock'], max_extra_csrs=case['max_extra_csrs'],
                selected_area_proxy=selection['selected_area'],
                software_model_cycles=selection['software_only_latency'],
                selected_model_cycles=selection['selected_latency'],
                normalized_software_baseline='passed; zero CIs')
            (native / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


TIMING = r'''
extern "C" volatile uint64_t tohost, fromhost;
static uint32_t cycles() {
 uint32_t x;
 asm volatile("fence rw,rw; csrr %0,mcycle" : "=r"(x) :: "memory");
 return x;
}
static void print_cycles(uint32_t cold, uint32_t warm) {
 char text[96]; int n=0;
 const char *label="APS_CYCLES cold=";
 while(*label) text[n++]=*label++;
 uint32_t counts[2]={cold,warm};
 for(int j=0;j<2;++j) {
  char digits[10]; int k=0; uint32_t x=counts[j];
  do {digits[k++]='0'+x%10; x/=10;} while(x);
  while(k) text[n++]=digits[--k];
  if(!j) {label=" warm=";while(*label)text[n++]=*label++;}
 }
 text[n++]='\n';
 alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
 asm volatile("fence rw,rw" ::: "memory");
 tohost=(uintptr_t)request;
 while(!fromhost) {}
 fromhost=0;
 asm volatile("fence rw,rw" ::: "memory");
}
'''

def instrument_harness(runtime, checked_values, boundary):
    """Time cold and warm calls while checking every output after each call."""
    declaration = 'extern "C" int example_main();'
    if runtime.count(declaration) != 1:
        raise ValueError('Expected one example_main declaration')
    runtime = runtime.replace(declaration, TIMING + '\nstatic uint32_t end_cycle;\n' + declaration)
    if boundary == 'observer':
        runtime, observers = re.subn(r'\+\+validation_calls;',
                                    'end_cycle=cycles();\n ++validation_calls;', runtime)
        pattern = r'int actual\s*=\s*example_main\(\);'
        cold_end = 'end_cycle'
        warm_end = 'end_cycle'
        validate = ''
        if observers != 1:
            raise ValueError('Expected one output observer')
    elif boundary == 'return':
        pattern = r'int actual\s*=\s*example_main\(\);\s*validate_outputs\(\);'
        cold_end = warm_end = 'cycles()'
        validate = 'validate_outputs();'
    else:
        raise ValueError(f'Unknown timing boundary: {boundary}')
    replacement = f"""uint32_t start_cycle=cycles();
 int actual=example_main();
 uint32_t cold={cold_end}-start_cycle;
 {validate}
 if(actual!=expected || mismatches)return 2;
 start_cycle=cycles();
 actual=example_main();
 uint32_t warm={warm_end}-start_cycle;
 {validate}
 print_cycles(cold,warm);"""
    runtime, calls = re.subn(pattern, replacement, runtime)
    runtime, checks = re.subn(
        r'if\s*\(\s*validation_calls\s*!=\s*1\s*\|\|\s*checked_outputs\s*!=[^\n;]+\)\s*return\s+3\s*;',
        f'if(validation_calls!=2 || checked_outputs!={2*checked_values}) return 3;', runtime)
    if (calls, checks) != (1, 1):
        raise ValueError('Expected one benchmark call and output-count check')
    return runtime


def measure_rocket(args):
    source, out = args.source_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((source/'summary.json').read_text())
    assert report['status'] == 'passed'
    profile = json.loads((source/'target-profile.json').read_text())
    if hashlib.sha256((source/'target-profile.json').read_bytes()).hexdigest() != report['target_profile_sha256']:
        raise RuntimeError('Target profile differs from the validated native run')
    max_extra_csrs = profile['interface']['max_extra_csrs']
    has_ci = report['ci_sites'] > 0
    soc = args.soc_dir.resolve() if args.soc_dir else out
    if not has_ci:
        raise RuntimeError('Rocket measurement requires a selected accelerator; native validation passed without one')
    executor = Commands(out, args.jobs)
    env = executor.env
    run = executor.run
    validation_harness = (args.harness or Path(report['validation_harness'])).resolve()
    checked_values = report['independently_checked_values_per_binary']
    runtime = validation_harness.read_text()
    runtime = re.sub(r'^#include "([^"]+)"',
                     lambda m: '#include "' + str((validation_harness.parent/m[1]).resolve()) + '"',
                     runtime, flags=re.MULTILINE)
    runtime = instrument_harness(runtime, checked_values, report['validation_boundary'])
    harness = out/'main.cpp'
    harness.write_text(runtime)
    for mode in ['baseline', 'accelerated']:
        mlir = source/('baseline.mlir' if mode == 'baseline' else 'selected.mlir')
        hardware = source/('hardware.mlir' if has_ci else 'baseline-hardware.mlir')
        run('compile-'+mode, [DRIVER, mlir, '--compile-only', f'--hardware={hardware}',
            *(['--allow-no-ci'] if mode == 'baseline' or not has_ci else []), *link_arguments(max_extra_csrs, harness), '-o', out/(mode+'.riscv')])
        compiled = json.loads((out/(mode+'.riscv.scaleisa/compile-report.json')).read_text())
        assert (compiled['ci_sites'] > 0) == (mode == 'accelerated' and has_ci)
    rtl = soc/'design.sv'
    if args.soc_dir:
        if not rtl.exists() or (has_ci and rtl.read_bytes() != (source/'design.sv').read_bytes()):
            raise RuntimeError('Reused SoC RTL does not match the selected accelerator')
    elif not rtl.exists() or rtl.read_bytes() != (source/'design.sv').read_bytes():
        shutil.copyfile(source/'design.sv', rtl)
    config = soc/'aps_config.yaml'
    pipeline = json.loads(Path(report['pipeline_report']).read_text())
    csr_arguments = pipeline.get('runtime_csr_arguments', []) if has_ci else []
    csrs = []
    for index, value in enumerate(csr_arguments):
        if value != '--add-csr':
            continue
        name, address = csr_arguments[index+1].rsplit('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise RuntimeError('Invalid generated CSR name')
        csrs.append(dict(name=name, id=int(address, 0)))
    csr_yaml = 'csrs: []\n' if not csrs else 'csrs:\n' + ''.join(
        f'  - id: {csr["id"]}\n    name: {csr["name"]}\n    mask: "0xffffffff"\n    init: 0\n'
        for csr in csrs)
    config_text = ('backend: rocc\narch: rv32\ntop_module: aps_rocc_wrapper\nname: aps_rocc_module\n'
                   'core_count: 1\nmaxBurstBytes: 128\nnXacts: 2\n' + csr_yaml + 'vsrc:\n'
                   f'  - "{rtl}"\n')
    if args.soc_dir and config.read_text() != config_text:
        raise RuntimeError('Reused SoC APS configuration differs from the benchmark configuration')
    if not args.soc_dir and (not config.exists() or config.read_text() != config_text):
        config.write_text(config_text)
    env['APS_CONFIG'] = str(config)
    make_dir = soc/'make-soc'
    simulator = soc/'simulator-debug'
    make_dir.mkdir(exist_ok=True)
    for name in ['bootrom.rv32.img', 'bootrom.rv64.img']:
        image = CHIPYARD/'generators/testchipip/bootrom'/name
        if not (make_dir/name).exists() or (make_dir/name).read_bytes() != image.read_bytes():
            shutil.copyfile(image, make_dir/name)
    make = ['make', '-C', CHIPYARD/'sims/verilator', 'CONFIG=APSRocketConfig',
            'run-binary-debug', 'LOADMEM=1', f'-j{args.jobs}', f'nproc={args.jobs}',
            f'build_dir={make_dir}', f'sim_debug={simulator}', f'output_dir={out/"output"}',
            'USE_FST=1', 'TRACING_OPTS=--trace-fst --trace-threads 1', 'VERBOSE_FLAGS=',
            'RANDOM_SEED=1', 'TIMEOUT_CYCLES=50000000',
            f'EXTRA_GENERATOR_REQS={config} {rtl} {make_dir/"bootrom.rv32.img"} {make_dir/"bootrom.rv64.img"}',
            f'EXTRA_SIM_REQS={rtl}']
    if not args.waveform:
        make.append('get_waveform_flag=')
    # Build once before launching independent simulations, avoiding concurrent
    # mutation of the generated SoC and Verilator build directories.
    run('build-chipyard', [arg for arg in make if arg != 'run-binary-debug']+[simulator])
    counts = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {mode: pool.submit(run, 'chipyard-'+mode,
                    make+[f'BINARY={out/(mode+".riscv")}']) for mode in ['baseline', 'accelerated']}
        for future in futures.values():
            future.result()
    for mode in ['baseline', 'accelerated']:
        log = (out/'output'/(mode+'.log')).read_text()
        match = re.search(r'APS_CYCLES cold=(\d+) warm=(\d+)', log)
        if not match:
            raise RuntimeError(f'Missing cycle counts for {mode}')
        counts[mode] = dict(cold=int(match[1]), warm=int(match[2]))
        (out/'partial-cycles.json').write_text(json.dumps(counts, indent=2)+'\n')
    summary = dict(status='passed', configuration='APSRocketConfig',
        processor='RV32 Rocket, full Chipyard Verilator',
        waveform_recording=args.waveform,
        chipyard_commit=subprocess.check_output(['git', '-C', CHIPYARD, 'rev-parse', 'HEAD'], text=True).strip(),
        measurement=('mcycle around the benchmark call; fences included; independent checking and printing excluded; boundary=' + report['validation_boundary']),
        correctness=f'All {checked_values} values and checksum checked on both cold and warm calls of both binaries',
        validation_harness=str(validation_harness),
        selection_contains_ci=has_ci,
        csr_bindings=csrs,
        max_extra_csrs=max_extra_csrs,
        cycles=counts, chipyard_speedup={state: counts['baseline'][state]/counts['accelerated'][state]
                                        for state in ['cold', 'warm']},
        scaleisa_estimated_speedup=report['estimated_program_speedup'],
        source_report=str(source/'summary.json'),
        scheduling_period_ns=report['scheduling_period_ns'],
        binary_sha256={mode: hashlib.sha256((out/(mode+'.riscv')).read_bytes()).hexdigest()
                       for mode in ['baseline', 'accelerated']},
        chipyard_wrapper_sha256=hashlib.sha256((CHIPYARD/'generators/aps/src/main/scala/ApsWrapper.scala').read_bytes()).hexdigest(),
        compiler_ci_lowering_sha256=hashlib.sha256((ROOT/'lib/scaleisa-compiler/LowerCI.cpp').read_bytes()).hexdigest(),
        rtl_sha256=hashlib.sha256(rtl.read_bytes()).hexdigest(),
        simulator_sha256=hashlib.sha256(simulator.read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)

def main() -> int:
    names = sorted(p.parent.name for p in (HERE / 'cases').glob('*/case.json'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=['all', *names], default=['all'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'build/aps-scaleisa-examples')
    parser.add_argument('--chipyard', action='store_true', help='Also measure both programs on full Rocket RTL')
    parser.add_argument('--jobs', type=int, help='Parallel jobs for driver and Chipyard stages')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Print commands without executing them')
    parser.add_argument('--prepare-case', choices=names, help=argparse.SUPPRESS)
    parser.add_argument('--measure-rocket', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--source-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--soc-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--harness', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--waveform', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs is not None and args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.prepare_case:
        prepare_case(load_case(args.prepare_case), args.output_dir.resolve(), args.jobs)
        return 0
    if args.measure_rocket:
        if args.source_dir is None:
            parser.error('--measure-rocket requires --source-dir')
        args.jobs = args.jobs or os.cpu_count() or 1
        measure_rocket(args)
        return 0
    if 'all' in args.cases and args.cases != ['all']:
        parser.error('use all by itself')
    selected = names if args.cases == ['all'] or args.list else list(dict.fromkeys(args.cases))
    cases = [load_case(name) for name in selected]
    if args.list:
        for case in cases:
            print(f"{case['id']:18} {case['name']}")
        return 0
    output = args.output_dir.resolve()
    if args.plan:
        plan = [{'id': case['id'], 'stages': command_plan(case, output, args.chipyard, args.jobs)}
                for case in cases]
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'chipyard': args.chipyard, 'cases': []}
    path = output / 'summary.json'
    for case in cases:
        row = {'id': case['id'], 'status': 'running'}
        summary['cases'].append(row)
        path.write_text(json.dumps(summary, indent=2) + '\n')
        try:
            run_case(case, output, args.chipyard, args.jobs)
        except (OSError, RuntimeError, ValueError) as error:
            row.update(status='failed', error=str(error))
            summary['status'] = 'failed'
            path.write_text(json.dumps(summary, indent=2) + '\n')
            print(f'{error}\nSee {path}', file=sys.stderr)
            return 1
        row['status'] = 'passed'
    summary['status'] = 'passed'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Passed: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
