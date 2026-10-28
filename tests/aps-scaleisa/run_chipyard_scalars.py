#!/usr/bin/env python3
"""Measure existing scalar cases using Chipyard's full Rocket Verilator flow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CHIPYARD = ROOT / 'thirdparty/chipyard'
DRIVER = ROOT / 'build/tools/aps-scaleisa-opt/aps-scaleisa-opt'

# HTIF output and validation execute after both timed calls. RV32 mcycle wraps
# safely under subtraction for these short workloads. Fences include outstanding
# memory operations; the identical counter/fence overhead remains in both runs.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', choices=['bitcount', 'sq_dist'],
                   default=['bitcount', 'sq_dist'])
    p.add_argument('--scalar-dir', type=Path, default=ROOT/'build/aps-scaleisa-scalars')
    p.add_argument('--output-dir', type=Path, default=ROOT/'build/aps-scaleisa-chipyard')
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--resynthesize', action=argparse.BooleanOptionalAction, default=True,
                   help='Regenerate accelerator RTL with the current APS lowering')
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prefix = ROOT/'.pixi/envs/default'
    env.update(RISCV=str(prefix/'riscv-tools'),
               LEGACY_FIRTOOL=str(ROOT/'thirdparty/firtool-1.75.0/bin/firtool'),
               JAVA_TOOL_OPTIONS=f'-Xmx8G -Xss8M -Djava.io.tmpdir={CHIPYARD}/.java_tmp -XX:ActiveProcessorCount={args.jobs}')
    env['PATH'] = f'{prefix}/riscv-tools/bin:{prefix}/bin:{env.get("PATH", "")}'
    env['LD_LIBRARY_PATH'] = f'{prefix}/lib:{prefix}/riscv-tools/lib'
    summary = dict(status='running', configuration='APSRocketConfig',
                   chipyard_commit=subprocess.check_output(['git', '-C', CHIPYARD, 'rev-parse', 'HEAD'], text=True).strip(),
                   processor='RV32 Rocket, full Chipyard Verilator',
                   measurement='mcycle around scalar_selected_main, including fences; validation and HTIF output excluded',
                   cold='first call after boot', warm='second call, same inputs',
                   speedup='software cycles / accelerated cycles on the same SoC simulator',
                   cases=[])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name in args.cases:
        source = args.scalar_dir.resolve()/name
        report = json.loads((source/'report.json').read_text())
        assert report['status'] == 'passed' and report['cosimulation'] == 'passed'
        profile = json.loads((source/'target-profile.json').read_text())
        clock = profile['hardware']['_profile']['clock_ns']
        max_extra_csrs = profile['interface']['max_extra_csrs']
        csrs = []
        arguments = report.get('runtime_csr_arguments', [])
        for index, argument in enumerate(arguments):
            if argument != '--add-csr':
                continue
            csr_name, address = arguments[index+1].rsplit('=', 1)
            if not re.fullmatch(r'[A-Za-z0-9_]+', csr_name):
                raise RuntimeError('Invalid generated CSR name')
            csrs.append(dict(name=csr_name, id=int(address, 0)))
        summary['scaleisa_cpu_model'] = report['selection']['software_model']
        d = output/name
        d.mkdir(exist_ok=True)
        commands = []

        def run(stage, command, cwd=ROOT):
            print(f'{name}: {stage}', flush=True)
            start = time.monotonic()
            with (d/(stage+'.log')).open('w') as log:
                result = subprocess.run(list(map(str, command)), cwd=cwd, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=3600)
            commands.append(dict(stage=stage, argv=list(map(str, command)), cwd=str(cwd),
                                 returncode=result.returncode, seconds=time.monotonic()-start))
            (d/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f'{stage}: see {d/(stage+".log")}')

        input_source = Path(report['input'])
        runtime = (input_source.parent/'main.cpp').read_text()
        runtime = runtime.replace('extern "C" int main(){', TIMING+'\nextern "C" int main(){')
        old = 'int result=scalar_selected_main();\n scalar_validate();'
        assert old in runtime
        runtime = runtime.replace(old, '''uint32_t t0=cycles();
 int result=scalar_selected_main();
 uint32_t cold=cycles()-t0;
 scalar_validate();
 t0=cycles();
 result|=scalar_selected_main();
 uint32_t warm=cycles()-t0;
 scalar_validate();
 print_cycles(cold,warm);''')
        samples = json.loads((input_source.parent/'provenance.json').read_text())['samples']
        runtime = runtime.replace('validation_calls!=1', 'validation_calls!=2')
        runtime = runtime.replace(f'checked_outputs!={samples}', f'checked_outputs!={2*samples}')
        harness = d/'main.cpp'
        harness.write_text(runtime)
        for mode, mlir in [('accelerated', source/'selected.mlir'), ('baseline', source/'baseline.mlir')]:
            command = [DRIVER, mlir, '--compile-only', f'--hardware={source/"hardware.mlir"}',
                       f'--max-extra-csrs={max_extra_csrs}',
                       '--emit=elf', '--entry=_start', '--abi=ilp32f',
                       '--mattr=+m,+a,+f,+zicsr,+zifencei', f'--link-source={harness}',
                       f'--link-source={ROOT/"tests/scaleisa-compiler/fixtures/start.S"}',
                       f'--link-arg=-T{ROOT/"tests/scaleisa-compiler/fixtures/ram.ld"}',
                       '-o', d/(mode+'.riscv')]
            command += ['--rename-main=scalar_selected_main'] if mode == 'accelerated' else ['--allow-no-ci']
            run('compile-'+mode, command)
            compiled = json.loads((d/(mode+'.riscv.scaleisa/compile-report.json')).read_text())
            assert (compiled['ci_sites'] > 0) == (mode == 'accelerated')
        rtl = source/'design.sv'
        if args.resynthesize:
            rtl = d/'design.sv'
            run('synthesize', [ROOT/'build/tools/aps-opt/aps-e2e', '--clock', str(clock),
                              '-i', source/'hardware.mlir', '--resource', source/'resource.json',
                              '--export', 'sv', '--dump-cmt2', d/'design.cmt2.mlir', '-o', rtl])
        config = d/'aps_config.yaml'
        csr_yaml = 'csrs: []\n' if not csrs else 'csrs:\n' + ''.join(
            f'  - id: {csr["id"]}\n    name: {csr["name"]}\n    mask: "0xffffffff"\n    init: 0\n'
            for csr in csrs)
        config_text = ('backend: rocc\narch: rv32\ntop_module: aps_rocc_wrapper\n'
                          'name: aps_rocc_module\ncore_count: 1\nmaxBurstBytes: 128\n'
                          f'nXacts: 2\n' + csr_yaml + f'vsrc:\n  - "{rtl}"\n')
        if not config.exists() or config.read_text() != config_text:
            config.write_text(config_text)
        env['APS_CONFIG'] = str(config)
        make_dir = d/'make-soc'
        make_dir.mkdir(exist_ok=True)
        for image_name in ['bootrom.rv32.img', 'bootrom.rv64.img']:
            image = CHIPYARD/'generators/testchipip/bootrom'/image_name
            if not (make_dir/image_name).exists() or (make_dir/image_name).read_bytes() != image.read_bytes():
                shutil.copyfile(image, make_dir/image_name)
        make = ['make', '-C', CHIPYARD/'sims/verilator', 'CONFIG=APSRocketConfig',
                'run-binary-debug', 'LOADMEM=1', f'-j{args.jobs}', f'nproc={args.jobs}',
                f'build_dir={d/"make-soc"}', f'sim_debug={d/"simulator-debug"}',
                f'output_dir={d/"output"}', 'USE_FST=1',
                'TRACING_OPTS=--trace-fst --trace-threads 1', 'VERBOSE_FLAGS=',
                'RANDOM_SEED=1', 'TIMEOUT_CYCLES=1000000',
                'get_waveform_flag=',
                f'EXTRA_GENERATOR_REQS={config} {rtl} {d/"make-soc/bootrom.rv32.img"} {d/"make-soc/bootrom.rv64.img"}']
        make.append(f'EXTRA_SIM_REQS={rtl}')
        counts = {}
        for mode in ['baseline', 'accelerated']:
            run('chipyard-'+mode, make+[f'BINARY={d/(mode+".riscv")}'])
            log = (d/'output'/(mode+'.log')).read_text()
            match = re.search(r'APS_CYCLES cold=(\d+) warm=(\d+)', log)
            if not match:
                raise RuntimeError(f'Missing target cycle counts in {d/"output"/(mode+".log")}')
            counts[mode] = dict(cold=int(match[1]), warm=int(match[2]))
        selection = report['selection']
        case = dict(name=name, status='passed', samples_per_call=samples,
                    validated_outputs_per_binary=2*samples, cycles=counts,
                    chipyard_speedup={state: counts['baseline'][state]/counts['accelerated'][state]
                                      for state in ['cold', 'warm']},
                    scaleisa_software_cycles=selection['software_only_latency'],
                    scaleisa_selected_cycles=selection['selected_latency'],
                    scaleisa_estimated_speedup=selection['software_only_latency']/selection['selected_latency'],
                    scheduling_period_ns=clock, max_extra_csrs=max_extra_csrs,
                    csr_bindings=csrs,
                    source_report=str(source/'report.json'),
                    binary_sha256={mode: hashlib.sha256((d/(mode+'.riscv')).read_bytes()).hexdigest()
                                   for mode in ['baseline', 'accelerated']},
                    selected_area=selection['selected_area'],
                    rtl=str(rtl), rtl_sha256=hashlib.sha256(rtl.read_bytes()).hexdigest(),
                    simulator_sha256=hashlib.sha256((d/'simulator-debug').read_bytes()).hexdigest())
        summary['cases'].append(case)
        (d/'result.json').write_text(json.dumps(case, indent=2)+'\n')
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(case), flush=True)
    summary['status'] = 'passed'
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(output/'summary.json')


if __name__ == '__main__':
    main()
