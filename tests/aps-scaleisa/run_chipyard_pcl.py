#!/usr/bin/env python3
"""Measure a validated PCL selection and CPU baseline on full Rocket RTL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from run_chipyard_scalars import TIMING, ROOT, CHIPYARD, DRIVER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, default=ROOT/'build/aps-octree-asap7-8ns-a5000')
    p.add_argument('--output-dir', type=Path, default=ROOT/'build/aps-octree-chipyard')
    p.add_argument('--harness', type=Path,
                   help='Output-validation harness; defaults to the source report or octree fixture')
    p.add_argument('--soc-dir', type=Path,
                   help='Reuse a built SoC; selected accelerator RTL must match for CI cases')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--waveform', action='store_true', help='Record full waveforms during measurement')
    args = p.parse_args()
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
    if not has_ci and not args.soc_dir:
        raise RuntimeError('Software-only selection requires --soc-dir pointing to an existing SoC')
    prefix = ROOT/'.pixi/envs/default'
    env = os.environ.copy()
    env.update(RISCV=str(prefix/'riscv-tools'),
               LEGACY_FIRTOOL=str(ROOT/'thirdparty/firtool-1.75.0/bin/firtool'),
               JAVA_TOOL_OPTIONS=f'-Xmx8G -Xss8M -Djava.io.tmpdir={CHIPYARD}/.java_tmp -XX:ActiveProcessorCount={args.jobs}')
    env['PATH'] = f'{prefix}/riscv-tools/bin:{prefix}/bin:{env.get("PATH", "")}'
    env['LD_LIBRARY_PATH'] = f'{prefix}/lib:{prefix}/riscv-tools/lib'
    commands = []
    command_lock = threading.Lock()

    def run(stage, command):
        print(stage, flush=True)
        start = time.monotonic()
        with (out/(stage+'.log')).open('w') as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        with command_lock:
            commands.append(dict(stage=stage, argv=list(map(str, command)), returncode=result.returncode,
                                 seconds=time.monotonic()-start))
            (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{stage} failed: {out/(stage+".log")}')

    validation_harness = (args.harness or Path(report.get('validation_harness',
        ROOT/'tests/aps-scaleisa/fixtures/octree_main.cpp'))).resolve()
    checked_values = report['independently_checked_values_per_binary']
    runtime = validation_harness.read_text()
    runtime = re.sub(r'^#include "([^"]+)"',
                     lambda m: '#include "' + str((validation_harness.parent/m[1]).resolve()) + '"',
                     runtime, flags=re.MULTILINE)
    runtime = runtime.replace('extern "C" int pcl_selected_main();',
                              TIMING+'\nstatic uint32_t end_cycle;\nextern "C" int pcl_selected_main();')
    # The observer is called immediately before main returns. Record its entry,
    # excluding all independent checking from the timed region in both binaries.
    runtime, observers = re.subn(r'\+\+validation_calls;',
                                'end_cycle=cycles();\n ++validation_calls;', runtime)
    runtime, calls = re.subn(r'int actual\s*=\s*pcl_selected_main\(\);', ''' uint32_t start_cycle=cycles();
 int actual=pcl_selected_main();
 uint32_t cold=end_cycle-start_cycle;
 if(actual!=expected || mismatches)return 2;
 start_cycle=cycles();
 actual=pcl_selected_main();
 uint32_t warm=end_cycle-start_cycle;
 print_cycles(cold,warm);''', runtime)
    runtime, checks = re.subn(
        r'if\s*\(\s*validation_calls\s*!=\s*1\s*\|\|\s*checked_outputs\s*!=[^\n;]+\)\s*return\s+3\s*;',
        f'if(validation_calls!=2 || checked_outputs!={2*checked_values}) return 3;', runtime)
    if (observers, calls, checks) != (1, 1, 1):
        raise RuntimeError('Validation harness must contain one observer, benchmark call and output-count check')
    harness = out/'main.cpp'
    harness.write_text(runtime)
    for mode in ['baseline', 'accelerated']:
        mlir = source/('baseline.mlir' if mode == 'baseline' else 'selected.mlir')
        hardware = source/('hardware.mlir' if has_ci else 'baseline-hardware.mlir')
        run('compile-'+mode, [DRIVER, mlir, '--compile-only', f'--hardware={hardware}',
            *(['--allow-no-ci'] if mode == 'baseline' or not has_ci else []), '--emit=elf', '--entry=_start',
            f'--max-extra-csrs={max_extra_csrs}',
            '--abi=ilp32f', '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--link-source={harness}',
            f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
            f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}', '-o', out/(mode+'.riscv')])
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
        measurement='mcycle from immediately before the benchmark call to output observer entry; fences included; reference generation, validation, HTIF printing and return epilogue excluded',
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


if __name__ == '__main__':
    main()
