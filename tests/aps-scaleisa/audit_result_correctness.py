#!/usr/bin/env python3
"""Replay result comparisons and one-bit wrong-answer controls on full Rocket RTL.

Reuse the measured hardware and program MLIR. Only the validation harness changes:
print executed comparison counts, or flip one bit of an actual output immediately
before the original element-wise software-reference comparisons.
"""
import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TARGET_ARGUMENT = {'octree': 'scores', 'iss': 'keypoints', 'normal': 'curvature',
                   'morphological': 'out', 'knn': 'idx', 'icp': 'residuals',
                   'ransac': 'buffer_2', 'segmentation': 'buffer_2',
                   'poisson': 'buffer_3', 'sfm_batch': 'buffer_0',
                   'stereo_vo': 'buffer_0', 'planar_slam': 'buffer_0',
                   'pose2_slam': 'buffer_4', 'pose3_pose_graph': 'buffer_0'}
REPORT = r'''
static void audit_report() {
 char text[128]; int n=0;
 const char *label="APS_RESULT_CHECK checked/mismatches/first=";
 while(*label)text[n++]=*label++;
 uint32_t values[3]={(uint32_t)checked_outputs,(uint32_t)mismatches,(uint32_t)first_mismatch};
 const char *hex="0123456789abcdef";
 for(int j=0;j<3;++j) {
  for(int k=7;k>=0;--k)text[n++]=hex[(values[j]>>(k*4))&15];
  text[n++]=j==2?'\n':'/';
 }
 alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
 asm volatile("fence rw,rw" ::: "memory");
 tohost=(uintptr_t)request;
 while(!fromhost){}
 fromhost=0;
 asm volatile("fence rw,rw" ::: "memory");
}
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument(source, case, corrupt):
    name = 'scalar_validate' if case in ('bitcount', 'sq_dist') else 'pcl_validate_outputs'
    match = re.search(r'\bvoid\s+' + name + r'\([^)]*\)\s*\{', source)
    assert match, case
    start = match.end()
    end, depth = start, 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    body = source[start:end-1]
    if corrupt:
        target = 'scalar_results' if name == 'scalar_validate' else TARGET_ARGUMENT[case]
        # Volatile byte accesses ensure the simulated computed output is changed.
        body = f'\n ((volatile unsigned char *){target})[0] ^= 1;\n' + body
    body += '\n audit_report();\n'
    return ('static void audit_report();\n' + source[:start] + body + source[end-1:] + REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', help='Audit only these manifest IDs')
    parser.add_argument('--reuse-passed', action='store_true')
    args = parser.parse_args()
    bundle = ROOT / 'reports/aps-scaleisa-16'
    out = ROOT / 'build/aps-suite-16/result-audit'
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    prefix = ROOT / '.pixi/envs/default'
    env['PATH'] = f'{prefix}/riscv-tools/bin:{prefix}/bin:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = f'{prefix}/lib:{prefix}/riscv-tools/lib'
    cases = json.loads((bundle / 'summary.json').read_text())['cases']
    if args.cases:
        assert set(args.cases) <= {row['id'] for row in cases}
        cases = [row for row in cases if row['id'] in args.cases]

    def audit(row):
        case = row['id']
        original = ROOT / row['chipyard']
        measured = json.loads(original.read_text())
        work = original.parent
        assert sha(work/'accelerated.riscv') == measured['binary_sha256']['accelerated']
        commands = json.loads((work/'commands.json').read_text())
        compile_command = next(c['argv'] for c in commands if c['stage'] == 'compile-accelerated')
        d = out / case
        d.mkdir(exist_ok=True)
        if args.reuse_passed and (d/'summary.json').exists():
            cached = json.loads((d/'summary.json').read_text())
            assert cached['status'] == 'passed'
            for test in cached['tests']:
                assert sha(d/(test['mode']+'.cpp')) == test['harness_sha256']
                assert sha(d/(test['mode']+'.riscv')) == test['binary_sha256']
                assert sha(work/'simulator-debug') == test['simulator_sha256'] == measured['simulator_sha256']
            return cached
        source = (work/'main.cpp').read_text()
        records = []
        for corrupt in (False, True):
            mode = 'corrupt-output' if corrupt else 'correct-output'
            harness = d / (mode+'.cpp')
            harness.write_text(instrument(source, case, corrupt))
            binary = d / (mode+'.riscv')
            command = []
            for arg in compile_command:
                if str(arg).startswith('--link-source=') and str(arg).endswith('/main.cpp'):
                    arg = '--link-source='+str(harness)
                elif str(arg) == str(work/'accelerated.riscv'):
                    arg = str(binary)
                command.append(str(arg))
            with (d/(mode+'-compile.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            simulator = work/'simulator-debug'
            assert sha(simulator) == measured['simulator_sha256']
            command = [str(simulator), '+permissive', '+dramsim',
                       '+dramsim_ini_dir='+str(ROOT/'thirdparty/chipyard/generators/testchipip/src/main/resources/dramsim2_ini'),
                       '+max-cycles=50000000', '+verilator+seed+1',
                       '+loadmem='+str(binary), '+permissive-off', str(binary)]
            logpath = d/(mode+'.log')
            with logpath.open('w') as log:
                result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            observations = [tuple(int(v,16) for v in match) for match in re.findall(
                r'APS_RESULT_CHECK checked/mismatches/first=([0-9a-f]{8})/([0-9a-f]{8})/([0-9a-f]{8})',logpath.read_text())]
            expected = measured.get('validated_outputs_per_binary')
            if expected is None:
                expected = 2*int(re.search(r'All (\d+) values', measured['correctness'])[1])
            if corrupt:
                assert result.returncode != 0 and observations and observations[0][1] == 1, (case,mode,result.returncode,observations)
                assert observations[0][2] != 0xffffffff
            else:
                assert result.returncode == 0 and len(observations) == 2, (case,mode,result.returncode,observations)
                assert observations[-1][0] == expected
                assert all(x[1] == 0 and x[2] == 0xffffffff for x in observations)
            records.append(dict(mode=mode,status='passed',returncode=result.returncode,
                                observations=[dict(checked=x[0],mismatches=x[1],first_mismatch=x[2]) for x in observations],
                                harness_sha256=sha(harness),binary_sha256=sha(binary),simulator_sha256=sha(simulator)))
            print(case,mode,'passed',observations,flush=True)
        receipt=dict(id=case,status='passed',tests=records)
        (d/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return receipt

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(audit,cases))
    summary=dict(status='passed',distinct_cases=len(results),cases=results,
                 scope='Measured inputs; same selected MLIR and full Chipyard hardware. Element-wise bit-exact software-reference comparisons; one-bit actual-output corruption must be rejected.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (bundle/'result-correctness.json').write_text(json.dumps(summary,indent=2)+'\n')
    report = ['# Numerical result correctness', '',
              'Each accelerated program was replayed on its full Chipyard Rocket RTL simulator. '
              'The original element-wise comparisons against independent C++ software references '
              'were instrumented to print executed comparison counts. Integers and float bit '
              'patterns must match exactly. These checks cover the measured deterministic inputs.', '',
              'A separate negative control flips one bit of an actual result buffer immediately '
              'before those comparisons. The unmodified checker must report one mismatch and '
              'the simulated program must fail. This changes the computed data, not the checker '
              'or its expected software answer. Original benchmark MLIR and RTL are reused.', '',
              '| Case | Compared words, two calls | Correct output | One-bit wrong output | Evidence |',
              '|---|---:|---|---|---|']
    for receipt in results:
        case = receipt['id']
        target = bundle/case/'correctness'
        target.mkdir(exist_ok=True)
        for name in ('correct-output.cpp', 'corrupt-output.cpp', 'correct-output.log',
                     'corrupt-output.log', 'summary.json'):
            shutil.copyfile(out/case/name, target/name)
        count = receipt['tests'][0]['observations'][-1]['checked']
        report.append(f'| {case} | {count:,} | Exact match | Rejected | '
                      f'[correct]({case}/correctness/correct-output.log), '
                      f'[corrupt]({case}/correctness/corrupt-output.log), '
                      f'[harness]({case}/correctness/corrupt-output.cpp) |')
    (bundle/'correctness.md').write_text('\n'.join(report)+'\n')


if __name__ == '__main__':
    main()
