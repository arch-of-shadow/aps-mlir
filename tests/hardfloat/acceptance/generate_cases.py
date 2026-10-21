#!/usr/bin/env python3
"""Materialize auditable CADL + C acceptance programs and exact bit goldens."""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ieee_oracle import IEEE

ROOT=Path(__file__).resolve().parents[3]
OUTPUT=ROOT/'build/hardfloat/acceptance/generated'
OPS=['add','sub','mul','div','sqrt','fma','fmsub','fnmadd','fnmsub','from_i32','from_u32','to_i32','to_u32','neg','abs','copysign','min','max','classify']
ALL=['add','sub','mul','div','sqrt','fma','cmp','from_i32','from_u32','to_i32','to_u32','widen','narrow','neg','abs','copysign','min','max','classify']

def special(w):
    o=IEEE(w)
    return [0,o.signbit,1,2,(1<<o.f)-1,1<<o.f,(127<<o.f), (127<<o.f)|o.signbit,
      (126<<o.f), (128<<o.f), (150<<o.f),(158<<o.f),o.max,o.max|o.signbit,
      o.inf,o.inf|o.signbit,o.nan,o.inf|1,o.nan|3,o.signbit|o.inf|1]


def make_case(precision,name,expressions,evaluate,variables=None,declarations=''):
    w=16 if precision=='bf16' else 32
    variables=variables or f'    let ab: u{w} = _irf[rs1];\n    let bb: u{w} = _irf[rs2];\n    let a: {"bf16" if w==16 else "f32"} = bitcast<{"bf16" if w==16 else "f32"}>(ab);\n    let b: {"bf16" if w==16 else "f32"} = bitcast<{"bf16" if w==16 else "f32"}>(bb);\n'
    base=f'{precision}_{name}'
    code=declarations+'#[opcode(7\'b0001011)]\n#[funct7(7\'b0000000)]\nrtype '+base+'(rs1:u5, rs2:u5, rd:u5) {\n'+variables+expressions+'\n}\n'
    (OUTPUT/(base+'.cadl')).write_text(code)
    (OUTPUT/(base+'.c')).write_text(f'''/* Paired with {base}.cadl; numerical reference: exact rational IEEE oracle. */
#include "common.h"
int run_test(const char *vectors) {{ return check_vectors("{base}", vectors); }}
''')
    rng=random.Random(0x48460000+w+sum(map(ord,name)))
    pairs=list(itertools.product(special(w),repeat=2))
    # Integer and cross-format inputs must also exercise all 32 input bits.
    pairs += [(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(1000)]
    return dict(name=base,width=w,pairs=pairs,evaluate=evaluate)


def main():
    global OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    OUTPUT=parser.parse_args().output.resolve()
    out=OUTPUT/'vectors';out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for precision,w in [('bf16',16),('fp32',32)]:
        ty='bf16' if w==16 else 'f32'; o=IEEE(w); mask=(1<<w)-1
        def invoke(op,flags=False,rm=0,pred=0):
            args='a,b,b' if op in ('fma','fmsub','fnmadd','fnmsub') else 'a,b' if op in ('add','sub','mul','div','cmp','copysign','min','max') else 'a'
            if op in ('from_i32','from_u32'):args='_irf[rs1]'
            if op=='cmp':args+=f',{pred}'
            elif op not in ('neg','abs','copysign','min','max','classify'):args+=f',{rm}'
            call=f'{precision}_{op}'+('_flags' if flags else '')+'('+args+')'
            return call if flags or op in ('cmp','classify','to_i32','to_u32') else f'bitcast<u{w}>({call})'
        for op in OPS:
            rm=1 if op in ('to_i32','to_u32') else 0
            code='    let result: u32 = '+invoke(op,rm=rm)+';\n    _irf[rd] = result;'
            def ref(a,b,op=op,rm=rm,o=o,mask=mask):
                if op in ('fmsub','fnmadd','fnmsub'):
                    aa=(a&mask)^(o.signbit if op in ('fnmadd','fnmsub') else 0)
                    cc=(b&mask)^(o.signbit if op in ('fmsub','fnmadd') else 0)
                    return o.operation('fma',aa,b&mask,cc,rm)[0]
                return o.operation(op,a if op.startswith('from_') else a&mask,b&mask,b&mask,rm)[0]
            cases.append(make_case(precision,op,code,ref))
        for flags in (False,True):
            decls='\n'.join(f'    let v{i}: u32 = {invoke("cmp",flags,pred=i)};' for i in range(16))
            expr=' | '.join(f'((v{i}'+(' >> 4)' if flags else ')')+f' << {i})' for i in range(16))
            def ref(a,b,flags=flags,o=o,mask=mask):
                return sum(((o.operation('cmp',a&mask,b&mask,pred=p)[1]>>4) if flags else o.operation('cmp',a&mask,b&mask,pred=p)[0])<<p for p in range(16))
            cases.append(make_case(precision,'cmp_flags' if flags else 'cmp',decls+f'\n    _irf[rd] = {expr};',ref))
        # Six independently packed 5-bit flag sets: no flags are silently dropped.
        flag_ops=['add','sub','mul','div','sqrt','fma']
        code='\n'.join(f'    let v{i}: u32 = {invoke(op,True)};' for i,op in enumerate(flag_ops))+'\n    _irf[rd] = '+' | '.join(f'(v{i} << {5*i})' for i,op in enumerate(flag_ops))+';'
        def ref(a,b,o=o,mask=mask):return sum(o.operation(op,a&mask,b&mask,b&mask)[1]<<(5*i) for i,op in enumerate(flag_ops))
        cases.append(make_case(precision,'flags',code,ref))
        misc_ops=['from_i32','from_u32','to_i32','to_u32','min','max']
        code='\n'.join(f'    let v{i}: u32 = {invoke(op,True)};' for i,op in enumerate(misc_ops))+'\n    _irf[rd] = '+' | '.join(f'(v{i} << {5*i})' for i,op in enumerate(misc_ops))+';'
        def ref(a,b,o=o,mask=mask):return sum(o.operation(op,a if op.startswith('from_') else a&mask,b&mask,rm=0)[1]<<(5*i) for i,op in enumerate(misc_ops))
        cases.append(make_case(precision,'misc_flags',code,ref))
        # All five RMs are also exercised through the actual CADL compiler.
        # Data digest here supplements full data+flags per-request IP verification.
        for op in ('add','mul','fma','to_i32','to_u32','from_i32','from_u32'):
            code='\n'.join(f'    let v{r}: u32 = {invoke(op,rm=r)};' for r in range(5))+'\n    _irf[rd] = '+' ^ '.join(f'(v{r} << {r})' for r in range(5))+';'
            def ref(a,b,op=op,o=o,mask=mask):
                v=0
                for r in range(5):v ^= o.operation(op,a if op.startswith('from_') else a&mask,b&mask,b&mask,r)[0]<<r
                return v&0xffffffff
            cases.append(make_case(precision,op+'_rounding',code,ref))
        # True float loop-carried values and branch captures, not integer emulation.
        loop=f'    with i: u32 = (0, i_) acc: {ty} = (a, acc_) do {{\n      let acc_: {ty} = acc + b;\n      let i_: u32 = i + 1;\n    }} while (i_ < 3);\n    let result: u32 = bitcast<u{w}>(acc);\n    _irf[rd] = result;'
        def ref(a,b,o=o,mask=mask):
            x=a&mask
            for i in range(3):x=o.operation('add',x,b&mask)[0]
            return x
        cases.append(make_case(precision,'loop',loop,ref))
        branch=f'    let r: {ty} = a;\n    if (_irf[rs1] & 32\'h80000000) != 0 {{ r = a + b; }} else {{ r = a - b; }}\n    let result: u32 = bitcast<u{w}>(r);\n    _irf[rd] = result;'
        def ref(a,b,o=o,mask=mask):return o.operation('add' if a&0x80000000 else 'sub',a&mask,b&mask)[0]
        cases.append(make_case(precision,'branch',branch,ref))
    for precision,op,w in [('bf16','widen',16),('fp32','narrow',32)]:
        outw=32 if op=='widen' else 16
        code=f'    let result: u32 = bitcast<u{outw}>({precision}_{op}(a));\n    _irf[rd] = result;'
        o=IEEE(w); mask=(1<<w)-1
        cases.append(make_case(precision,op,code,lambda a,b,o=o,op=op,mask=mask:o.operation(op,a&mask)[0]))
        code=f'    let result: u32 = {precision}_{op}_flags(a);\n    _irf[rd] = result;'
        cases.append(make_case(precision,op+'_flags',code,lambda a,b,o=o,op=op,mask=mask:o.operation(op,a&mask)[1]))
    for precision,w in [('bf16',16),('fp32',32)]:
        o=IEEE(w);mask=(1<<w)-1;ty='bf16' if w==16 else 'f32';bias=0x3fa0 if w==16 else 0x3fa00000
        decimal='1.00390625000000000000000000001' if w==16 else '1.000000059604644775390625000001'
        bits=0x3f81 if w==16 else 0x3f800001
        code=f'    let constant:{ty} = {decimal}_{ty};\n    let result:u32 = bitcast<u{w}>({precision}_add(a,constant));\n    _irf[rd] = result;'
        cases.append(make_case(precision,'constant',code,lambda a,b,o=o,mask=mask,bits=bits:o.operation('add',a&mask,bits)[0]))
        code=f'    let result:u32 = bitcast<u{w}>({precision}_fma(a,b,bias));\n    _irf[rd] = result;'
        cases.append(make_case(precision,'global_init',code,lambda a,b,o=o,mask=mask,bias=bias:o.operation('fma',a&mask,b&mask,bias)[0],declarations=f'static bias:{ty} = 1.25;\n'))
        code=f'    let index:u32 = ab & 1;\n    scratch[index] = a;\n    scratch[1-index] = b;\n    let x:{ty} = scratch[index];\n    let y:{ty} = scratch[1-index];\n    let result:u32 = bitcast<u{w}>({precision}_add(x,y));\n    _irf[rd] = result;'
        cases.append(make_case(precision,'memory',code,lambda a,b,o=o,mask=mask:o.operation('add',a&mask,b&mask)[0],declarations=f'static scratch:[{ty};2];\n'))
    o=IEEE(16)
    variables='    let ab:u16 = _irf[rs1];\n    let bb:u16 = _irf[rs1] >> 32\'h00000010;\n    let cb:u16 = _irf[rs2];\n    let a:bf16 = bitcast<bf16>(ab);\n    let b:bf16 = bitcast<bf16>(bb);\n    let c:bf16 = bitcast<bf16>(cb);\n'
    case=make_case('bf16','fma3','    let result:u32 = bitcast<u16>(bf16_fma(a,b,c));\n    _irf[rd] = result;',lambda a,b:o.operation('fma',a&65535,a>>16,b&65535)[0],variables)
    case['pairs'] += [(a|(b<<16),c) for a,b,c in itertools.product(special(16),repeat=3)]
    cases.append(case)
    manifest=[]
    for case in cases:
        path=out/(case['name']+'.txt')
        path.write_text(''.join(f'{a:08x} {b:08x} {case["evaluate"](a,b):08x}\n' for a,b in case['pairs']))
        manifest.append({k:v for k,v in case.items() if k not in ('pairs','evaluate')}|{'vectors':str(path.relative_to(OUTPUT)),'count':len(case['pairs'])})
    (OUTPUT/'cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'{len(manifest)} paired CADL/C cases; {sum(c["count"] for c in manifest)} ISAX reference requests')

if __name__=='__main__':main()
