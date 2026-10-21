#!/usr/bin/env python3
"""CADL/C -> CMT2 -> RTL -> IHP mapped gates -> gate simulation + real STA.
Numerical/protocol failures and STA errors are fatal. Timing slack is reported
separately; --require-timing additionally makes negative slack fatal.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PHYSICAL=ROOT/'hardware/hardfloat/synthesis'
sys.path.insert(0,str(PHYSICAL))
from physical_flow import synthesize as synthesize_design
OUT=ROOT/'build/hardfloat/acceptance'
GENERATED=OUT/'generated'
PDK=ROOT/'thirdparty/chipyard/nextvlsi/ihp13/pdk/ihp-sg13g2/libs.ref/sg13g2_stdcell'
LIB=PDK/'lib/sg13g2_stdcell_typ_1p20V_25C.lib'
CELLS=PDK/'verilog/sg13g2_stdcell.v'
STA=ROOT/'build/opensta/package/usr/bin/sta'
FUNCTIONAL_CELLS=OUT/'sg13g2_functional.v'
RUN_SIGNATURE=''
REQUIRE_TIMING=False
RTL_ONLY=False
import importlib.util
_spec=importlib.util.spec_from_file_location('hardfloat_package',ROOT/'hardware/hardfloat/build-hardfloat-ip.py')
_package=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_package)


def source_signature():
    paths=[HERE.parent/'sta.tcl',HERE.parent/'ieee_oracle.py',ROOT/'build/tools/aps-opt/aps-e2e',ROOT/'build/hardfloat/resource.json']
    for base,patterns in [(HERE,['*.py','*.cpp','*.h']),(GENERATED,['*.c','*.cadl','cases.json','vectors/*.txt']),(PHYSICAL,['*.py','*.abc']),(ROOT/'hardware/hardfloat',['*.sv','*.yaml']),(ROOT/'cadl_frontend',['*.py','*.lark']),(ROOT/'cadl_frontend/to_mlir',['*.py']),(ROOT/'hardware/hardfloat',['build-hardfloat-ip.py'])]:
        for pattern in patterns:paths.extend(base.glob(pattern))
    if not RTL_ONLY: paths.append(LIB)
    digest=hashlib.sha256()
    digest.update(f'rtl_only={RTL_ONLY}'.encode())
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args,log,cwd=ROOT,env=None):
    with Path(log).open('w') as f:
        result=subprocess.run([str(a) for a in args],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
    if result.returncode:raise RuntimeError(f'{Path(log).name}: exit {result.returncode}')


def prepare_cell_models():
    # The PDK simulation models depend on specify-driven delayed nets and UDP
    # event semantics unsupported by Verilator's zero-delay two-state mode.
    # Generate equivalent nonblocking functional models from the very same
    # unmodified Liberty used for mapping/STA. Never substitute ideal timing.
    temporary=FUNCTIONAL_CELLS.with_suffix(f'.{os.getpid()}.tmp')
    run(['yosys','-Q','-T','-p',f'read_liberty -ignore_miss_func {LIB}; write_verilog -noattr {temporary}'],OUT/f'cell-models-{os.getpid()}.log')
    temporary.replace(FUNCTIONAL_CELLS)


def bootstrap_sta():
    if STA.exists():return
    directory=STA.parents[3];directory.mkdir(parents=True,exist_ok=True)
    run(['apt','download','opensta'],directory/'download.log',cwd=directory)
    package=next(directory.glob('opensta*.deb'))
    expected='b1272a4816bc2c67098872f39820f27b4dddf640205e09a31444c71c9fe41e9e'
    if hashlib.sha256(package.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('OpenSTA package hash differs; review package provenance before using it')
    run(['dpkg-deb','-x',package,directory/'package'],directory/'extract.log')


def synthesize(directory,source,top,parameters=None):
    return synthesize_design(directory,source,top,LIB,parameters)


def timing(directory,netlist,top,clock,reset):
    env=dict(os.environ,LIBERTY=str(LIB),NETLIST=str(netlist),TOP=top,CLOCK_PORT=clock,RESET_PORT=reset)
    run([STA,'-no_splash','-exit',HERE.parent/'sta.tcl'],directory/'sta.log',cwd=directory,env=env)
    text=(directory/'sta.log').read_text()
    if 'Error:' in text or 'missing set_input_delay' in text or 'missing a clock' in text:
        raise RuntimeError('STA errors/unconstrained timing: sta.log')
    header=text.split('ACCEPTANCE_SETUP')[0]
    unconstrained=re.findall(r'^  (\S+)$',header,re.M)
    design=json.loads((directory/'buffered.json').read_text())['modules'][top]
    constants=set()
    for name,port in design['ports'].items():
        if port['direction']!='output':continue
        for i,bit in enumerate(port['bits']):
            if bit in ('0','1'):
                constants.add(name if len(port['bits'])==1 else f'{name}[{i+port.get("offset",0)}]')
    remaining=[n for n in unconstrained if n not in constants]
    if remaining:
        # Constant-fed DFF data pins also have no timing startpoint. Prove
        # those constants with the Liberty functional logic, never waive an
        # unconstrained real data path merely because it is internal.
        proof=directory/'constant-proof.json'
        run(['yosys','-Q','-T','-p',f'read_verilog {FUNCTIONAL_CELLS}; read_verilog {netlist}; hierarchy -top {top}; proc; flatten; opt -full; write_json {proof}'],directory/'constant-proof.log')
        nets=json.loads(proof.read_text())['modules'][top]['netnames']
        for endpoint in remaining:
            wire=endpoint.replace('/','.')
            net=nets.get(wire)
            if net and all(b in ('0','1') for b in net['bits']):constants.add(endpoint)
        remaining=[n for n in remaining if n not in constants]
        if remaining:raise RuntimeError('unconstrained nonconstant STA endpoints: '+str(remaining[:8]))
    setup=re.findall(r'worst slack\s+([-\d.]+)',text)
    hold=re.findall(r'([-\d.]+)\s+slack \((?:MET|VIOLATED)\)',text.split('ACCEPTANCE_HOLD')[-1])
    if not setup or not hold:raise RuntimeError('missing setup/hold reports')
    s,h=float(setup[-1]),min(map(float,hold))
    return dict(setup_slack_ns=s,hold_slack_ns=h,clock_ns=4.0,corner='TT 1.20V 25C',phase_pass=s>=0 and h>=0)


def compile_sim(directory,source,driver,top,extra=()):
    obj=directory/'obj'
    args=['verilator','--assert','--cc','--exe','--build','-j','2','-Wno-fatal','--top-module',top,'--Mdir',obj,'--output-split','12000','--output-split-cfuncs','3000','-CFLAGS','-O1',source]
    if source.name=='gates.v':args.append(FUNCTIONAL_CELLS)
    args.extend([driver,*extra])
    run(args,directory/'compile.log')
    return obj/('V'+top)


def case_test(case):
    name=case['name'];d=OUT/name;d.mkdir(parents=True,exist_ok=True)
    report={'name':name,'passed':False,'rtl_only':RTL_ONLY,'signature':RUN_SIGNATURE,'reference_requests':case['count']}
    try:
        source=d/'input.mlir';run([ROOT/'tools/aps-frontend','mlir',GENERATED/(name+'.cadl')],source)
        rtl=d/'rtl.sv';run([ROOT/'build/tools/aps-opt/aps-e2e','-i',source,'--resource',ROOT/'build/hardfloat/resource.json','--export','sv','-o',rtl],d/'codegen.log')
        cobj=d/'test.o';run(['cc','-std=c11','-O1','-I',HERE,'-c',GENERATED/(name+'.c'),'-o',cobj],d/'c-compile.log')
        native=d/'native';native.mkdir(exist_ok=True)
        exe=compile_sim(native,rtl,HERE/'driver.cpp','main',[cobj]);run([exe,GENERATED/case['vectors']],native/'run.log')
        report['rtl_pass']=True
        if RTL_ONLY:
            report['passed']=True
        else:
            gate=d/'mapped';gate.mkdir(exist_ok=True)
            net,buffers=synthesize(gate,rtl,'main');report['buffers']=buffers
            # Always test numerical correctness after physical pipeline construction.
            exe=compile_sim(gate,net,HERE/'driver.cpp','main',[cobj]);run([exe,GENERATED/case['vectors']],gate/'run.log')
            report['gate_pass']=True
            report['timing']=timing(gate,net,'main','clk','rst')
            report['timing_met']=report['timing']['phase_pass'];report['passed']=not REQUIRE_TIMING or report['timing_met']
            if not report['passed']:report['error']='setup/hold timing violation'
    except Exception as exc:report['error']=str(exc)
    (d/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(('PASS ' if report['passed'] else 'FAIL ')+name+(': '+report.get('error','') if not report['passed'] else ''),flush=True)
    return report


def unit_test(config):
    sys.path.insert(0,str(HERE.parent))
    from ieee_oracle import IEEE
    from generate_cases import ALL,special
    import itertools,random
    w,op,pred=config;name=f'ip_w{w}_{op}_p{pred}';d=OUT/name;d.mkdir(parents=True,exist_ok=True)
    report={'name':name,'passed':False,'rtl_only':RTL_ONLY,'signature':RUN_SIGNATURE}
    try:
        oracle=IEEE(w);rng=random.Random(0x163200+w+ALL.index(op))
        triples=[(a,b,special(w)[(i+j)%20]) for i,a in enumerate(special(w)) for j,b in enumerate(special(w))]
        triples += [tuple(rng.getrandbits(32 if op in ('from_i32','from_u32','narrow') else w) for _ in range(3)) for i in range(1000)]
        if w==16 and op in ('sqrt','classify','neg','abs','to_i32','to_u32','widen'):
            triples=[(a,0,0) for a in range(65536)]
        report['reference_requests']=len(triples)*5
        vectors=d/'vectors.txt'
        with vectors.open('w') as f:
            for a,b,c in triples:
                for rm in range(5):
                    value,flags=oracle.operation(op,a,b,c,rm,pred)
                    f.write(f'{a:08x} {b:08x} {c:08x} {rm:x} {value:08x} {flags:02x}\n')
        # Ensure the bundle includes current sources, using the same packager.
        run([ROOT/'hardware/hardfloat/build-hardfloat-ip.sh',f'width={w}',f'op={ALL.index(op)}',f'pred={pred}'],d/'package.log',cwd=d)
        source=ROOT/'build/hardfloat/rtl/HardFloatIP.sv'
        t=_package.native_timing(w,ALL.index(op));lat=t['latency_upper_bound'];cap=t['outstanding_capacity']
        parameters={'WIDTH':w,'OP':ALL.index(op),'PRED':pred,'LATENCY':lat,'CAPACITY':cap}
        native=d/'native';native.mkdir(exist_ok=True)
        obj=native/'obj'
        run(['verilator','--assert','--cc','--exe','--build','-j','2','-Wno-fatal','--top-module','HFUnit','--Mdir',obj,'-CFLAGS','-O1',*[f'-G{k}={v}' for k,v in parameters.items()],source,HERE/'unit_driver.cpp'],native/'compile.log')
        # The iterative wrapper has only one outstanding transaction, so its
        # ready latency can be checked even while collection is backpressured.
        metadata = json.loads((d/f'HardFloat_w{w}_o{ALL.index(op)}_p{pred}_l{lat}_q{cap}.json').read_text())
        bound = metadata['timing']['latency_upper_bound']
        test_args=[str(bound),str(int(t['variable_latency'])),str(cap)]
        run([obj/'VHFUnit',vectors,*test_args],native/'run.log');report['rtl_pass']=True
        if RTL_ONLY:
            report['passed']=True
        else:
            gate=d/'mapped';gate.mkdir(exist_ok=True)
            net,count=synthesize(gate,source,'HFUnit',parameters)
            exe=compile_sim(gate,net,HERE/'unit_driver.cpp','HFUnit');run([exe,vectors,*test_args],gate/'run.log')
            report['gate_pass']=True;report['buffers']=count
            report['timing']=timing(gate,net,'HFUnit','clock','reset');report['timing_met']=report['timing']['phase_pass'];report['passed']=not REQUIRE_TIMING or report['timing_met']
            if not report['passed']:report['error']='setup/hold timing violation'
    except Exception as exc:report['error']=str(exc)
    (d/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(('PASS ' if report['passed'] else 'FAIL ')+name+(': '+report.get('error','') if not report['passed'] else ''),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=3);p.add_argument('--only',default='');p.add_argument('--skip-passed',action='store_true');p.add_argument('--units',action='store_true');p.add_argument('--require-timing',action='store_true');p.add_argument('--rtl-only',action='store_true',help='Run compiler/RTL simulation without cell mapping or OpenSTA');args=p.parse_args()
    global REQUIRE_TIMING, RTL_ONLY
    if args.rtl_only and args.require_timing: p.error('--rtl-only cannot require timing')
    RTL_ONLY=args.rtl_only
    REQUIRE_TIMING=args.require_timing
    global RUN_SIGNATURE
    OUT.mkdir(parents=True,exist_ok=True)
    if not RTL_ONLY:
        bootstrap_sta();prepare_cell_models()
    RUN_SIGNATURE=source_signature()
    if args.units:
        from generate_cases import ALL
        cases=[(w,op,pred) for w in (16,32) for op in ALL if (op!='widen' or w==16) and (op!='narrow' or w==32) for pred in (range(16) if op=='cmp' else [0])]
        cases=[c for c in cases if re.search(args.only,f'ip_w{c[0]}_{c[1]}_p{c[2]}')]
        cached={}
        if args.skip_passed:
            for c in cases:
                result=OUT/f'ip_w{c[0]}_{c[1]}_p{c[2]}'/'result.json'
                if result.exists():
                    report=json.loads(result.read_text())
                    if report.get('passed') and (not REQUIRE_TIMING or report.get('timing_met')) and report.get('signature')==RUN_SIGNATURE:cached[c]=report
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:reports=list(cached.values())+list(pool.map(unit_test,[c for c in cases if c not in cached]))
        (OUT/'unit-summary.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(f'{sum(r["passed"] for r in reports)}/{len(reports)} IP functional configurations passed' + ('' if RTL_ONLY else f'; {sum(r.get("timing_met",False) for r in reports)}/{len(reports)} meet timing'))
        return 0 if reports and all(r['passed'] for r in reports) else 1
    cases=json.loads((GENERATED/'cases.json').read_text());cases=[c for c in cases if re.search(args.only,c['name'])]
    cached={}
    if args.skip_passed:
        for c in cases:
            result=OUT/c['name']/'result.json'
            if result.exists():
                report=json.loads(result.read_text())
                if report.get('passed') and (not REQUIRE_TIMING or report.get('timing_met')) and report.get('signature')==RUN_SIGNATURE:cached[c['name']]=report
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:reports=list(cached.values())+list(pool.map(case_test,[c for c in cases if c['name'] not in cached]))
    (OUT/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(f'{sum(r["passed"] for r in reports)}/{len(reports)} CADL/C functional acceptance passed' + ('' if RTL_ONLY else f'; {sum(r.get("timing_met",False) for r in reports)}/{len(reports)} meet timing'))
    return 0 if reports and all(r['passed'] for r in reports) else 1

if __name__=='__main__':sys.exit(main())
