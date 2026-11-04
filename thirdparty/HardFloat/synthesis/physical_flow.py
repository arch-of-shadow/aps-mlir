#!/usr/bin/env python3
"""Map native register boundaries to IHP SG13G2; never insert latency padding."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from buffer_flops import buffer_flops

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DEFAULT_LIB=ROOT/'thirdparty/chipyard/nextvlsi/ihp13/pdk/ihp-sg13g2/libs.ref/sg13g2_stdcell/lib/sg13g2_stdcell_typ_1p20V_25C.lib'


def run(args,log):
    with Path(log).open('w') as stream:
        result=subprocess.run([str(a) for a in args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{Path(log).name}: exit {result.returncode}')


def synthesize(directory,source,top,liberty=DEFAULT_LIB,parameters=None):
    directory=Path(directory).resolve();source=Path(source).resolve();liberty=Path(liberty).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    param=''
    if parameters:param='chparam '+ ' '.join(f'-set {k} {v}' for k,v in parameters.items())+' '+top+'; '
    # Parameterized source IP uses Yosys's deferred Verilog elaboration;
    # exported APS RTL needs Slang for automatic SystemVerilog local variables.
    reader = (f'read_verilog -sv -DSYNTHESIS {source}' if parameters else
              f'read_slang --top {top} -DSYNTHESIS {source}')
    command=f'read_liberty -lib {liberty}; {reader}; {param}hierarchy -check -top {top}; synth -top {top} -flatten -noabc; dffunmap; dfflibmap -liberty {liberty}; abc -liberty {liberty} -script {HERE / "map.abc"}; clean; check -assert; write_json {directory / "mapped.json"}'
    frontend = [] if parameters else ['-m', 'slang']
    run(['yosys','-Q','-T',*frontend,'-p',command],directory/'synthesis.log')
    mapped=json.loads((directory/'mapped.json').read_text())['modules'][top]
    cells=mapped['cells']
    info=dict(mode='native-register-boundaries', inserted_pipeline_stages=0,
              mapped_cells=len(cells), mapped_dffs=sum('dfr' in c['type'] or 'dff' in c['type'] for c in cells.values()))
    (directory/'pipeline.json').write_text(json.dumps(info,indent=2)+'\n')
    count=buffer_flops(directory/'mapped.json',directory/'buffered.json',top)
    netlist=directory/'gates.v'
    run(['yosys','-Q','-T','-p',f'read_json {directory / "buffered.json"}; hierarchy -top {top}; check -assert; stat -liberty {liberty}; splitnets; write_verilog -noattr -noexpr -simple-lhs {netlist}'],directory/'mapping.log')
    (directory/'provenance.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),liberty=str(liberty),liberty_sha256=hashlib.sha256(liberty.read_bytes()).hexdigest(),buffers=count),indent=2)+'\n')
    return netlist,count


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--top',default='main')
    parser.add_argument('--liberty',type=Path,default=DEFAULT_LIB)
    args=parser.parse_args()
    print(synthesize(args.output,args.input,args.top,args.liberty)[0])
