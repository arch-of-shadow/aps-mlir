"""Compiler and generated RTL regressions for elastic scratchpad accesses."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
E2E = ROOT / 'build/tools/aps-opt/aps-e2e'


def run(args, **kwargs):
    result = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                            text=True, **kwargs)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-6000:]
    return result


@pytest.fixture(scope='module')
def resource(tmp_path_factory):
    out = tmp_path_factory.mktemp('pipeline-resource')
    result = subprocess.run(['bash', str(ROOT / 'hardware/hardfloat/build-hardfloat-ip.sh'),
                             f'--resource-output={out / "resource.json"}'],
                            cwd=out, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return out / 'resource.json'


def compile_cadl(out, source, resource, export='sv'):
    cadl = out / 'input.cadl'
    cadl.write_text(source)
    mlir = out / 'input.mlir'
    run([ROOT / 'tools/aps-frontend', 'mlir', cadl, '-o', mlir])
    output = out / ('design.sv' if export == 'sv' else 'design.mlir')
    result = run([E2E, '-i', mlir, '--resource', resource, '--export', export,
         '--dump-cmt2', out / 'design.cmt2.mlir', '-o', output])
    (out / 'compile.log').write_text(result.stdout + result.stderr)
    return output


def test_read_channel_backpressure(tmp_path, resource):
    sv = compile_cadl(tmp_path, '''static data: [u32; 32];
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype reads(rs1:u5, rs2:u5, rd:u5) {
    let a:u32 = data[_irf[rs1]];
    let b:u32 = data[_irf[rs2]];
    _irf[rd] = a + b;
}
''', resource)
    run(['verilator', '--cc', '--exe', '--build', '-j', '3', '--assert',
         '-Wno-fatal', '--top-module', 'ScratchpadMemoryPool', '--Mdir',
         tmp_path / 'obj', sv, ROOT / 'tests/pipeline_memory/read_channels.cpp',
         '-CFLAGS', '-std=c++17 -O1'], timeout=180)
    for seed in [1, 7, 23]:
        run([tmp_path / 'obj/VScratchpadMemoryPool', seed], timeout=20)


@pytest.mark.parametrize('kind,ii', [('independent', 1), ('independent', 2),
                                    ('independent', 10), ('raw', 1),
                                    ('distance2', 1), ('dualread', 1), ('war_waw', 1)])
def test_pipeline_memory_order(tmp_path, resource, kind, ii):
    import re
    expr = {'independent': 'i + 1 + inc', 'raw': 'data[i] + inc',
            'distance2': 'data[i] + inc', 'dualread': 'data[i] + data[i + 1] + inc', 'war_waw': 'data[0] * 3 + inc'}[kind]
    dest = '0' if kind == 'war_waw' else ('i + 2' if kind in ('distance2', 'dualread') else 'i + 1')
    result = '0' if kind == 'war_waw' else ('n + 1' if kind in ('distance2', 'dualread') else 'n')
    sv = compile_cadl(tmp_path, f"""#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static data: [u32; 64];
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype loop_test(rs1:u5, rs2:u5, rd:u5) {{
    let n:u32 = _irf[rs1];
    let inc:u32 = _irf[rs2];
    data[0] = 0;
    data[1] = 0;
    [[pipeline(1)]]
    [[II({ii})]]
    with i:u32 = (0, next) do {{
        data[{dest}] = {expr};
        let next:u32 = i + 1;
    }} while (next < n);
    _irf[rd] = data[{result}];
}}
""", resource)
    text = sv.read_text()
    tokens = re.findall(r'(?:TokenFIFO(?:1_PUSH|2_I)\d*|FIFO(?:1_PUSH|2_I)_\w+) (op0b00_l\d+_b0_s(\d+)tok) \(', text)
    assert tokens, 'Missing pipelined stages'
    first = min(tokens, key=lambda p: int(p[1]))[0]
    achieved = re.findall(r'Pipeline scheduled II=(\d+)', (tmp_path / 'compile.log').read_text())
    assert achieved
    achieved = int(achieved[-1])
    monitor = tmp_path / 'monitor.sv'
    monitor.write_text(f"""
module AdmissionCheck(input clk, rst, admit);
 integer tick=0, last=-100000, tight=0;
 always @(posedge clk) begin
   tick=tick+1;
   if(rst)last=-100000;
   else if(admit)begin
     if(tick-last<{achieved})$fatal(1, "scheduled II violated");
     if(tick-last=={achieved})tight=tight+1;
     last=tick;
   end
 end
 final if({int(kind == "independent")} && tight==0)$fatal(1, "avoidable II bubbles");
endmodule
bind main AdmissionCheck admission_check(.clk(clk),.rst(rst),
 .admit({first}.enq_enable && {first}.enq_ready));
""")
    driver = tmp_path / 'driver.cpp'
    driver.write_text((ROOT / 'tests/hardfloat/test_isax.cpp').read_text()
                     .replace('next < vectors.size() && rng()',
                              'next < vectors.size() && pending.empty() && rng()')
                     .replace('vectors.size() * 500 + 1000', 'vectors.size() * 3000 + 1000'))
    vectors=[]
    for repeat in range(2):
        for n in [0, 1, 2, 3, 7, 16, 31, 48]:
            inc=17+repeat*41
            if kind == 'independent': expected=n+inc if n else 0
            elif kind == 'raw': expected=n*inc
            elif kind == 'distance2': expected=((n+1)//2)*inc
            elif kind == 'dualread':
                prev, expected = 0, 0
                for _ in range(n): prev, expected = expected, (prev+expected+inc)&0xffffffff
            else:
                expected=0
                for _ in range(n): expected=(expected*3+inc)&0xffffffff
            vectors.append(f'{n:x} {inc:x} {expected:x} 0')
    (tmp_path / 'vectors.txt').write_text('\n'.join(vectors)+'\n')
    run(['verilator', '--cc', '--exe', '--build', '-j', '3', '--assert',
         '-Wno-fatal', '--top-module', 'main', '--Mdir', tmp_path / 'obj',
         sv, monitor, driver, '-CFLAGS', '-std=c++17 -O1'], timeout=180)
    run([tmp_path / 'obj/Vmain', tmp_path / 'vectors.txt'], timeout=30)


@pytest.mark.parametrize('size', [4, 64])
@pytest.mark.parametrize('dtype,value', [('u32', '1'), ('bf16', '1.0_bf16'),
                                         ('f32', '1.0_f32')])
def test_partition_store_has_no_preservation_reads(tmp_path, resource, dtype, value, size):
    output = compile_cadl(tmp_path, f'''#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static data: [{dtype}; {size}];
#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype writes(rs1:u5, rs2:u5, rd:u5) {{
    [[pipeline(1)]]
    [[II(1)]]
    with i:u32 = (0, next) do {{
        data[i] = {value};
        let next:u32 = i + 1;
    }} while (next < {min(16, size)});
    _irf[rd] = 0;
}}
''', resource, export='cmt2')
    text = output.read_text()
    assert '@data_0_write_if(' in text and '@data_3_write_if(' in text
    assert 'capture_read' not in text, 'A write-only kernel must not read old bank contents'
    assert '_admit_cooldown' not in text  # scheduled II=1 needs no cooldown register
    run([ROOT / 'circt/build/bin/circt-opt', output,
         '--cmt2-inline-modules', '--cmt2-inline-private-funcs',
         '--cmt2-verify-private-funcs-inlined', '--cmt2-verify-call-sequence',
         '-o', tmp_path / 'verified.mlir'], timeout=120)


def test_scalar_recurrence_is_rejected_explicitly(tmp_path, resource):
    cadl = tmp_path / 'recurrence.cadl'
    cadl.write_text('''#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype recurrence(rs1:u5, rs2:u5, rd:u5) {
    let n:u32 = _irf[rs1];
    [[pipeline(1)]]
    [[II(1)]]
    with i:u32 = (0, next) acc:u32 = (0, sum) do {
        let sum:u32 = acc + i;
        let next:u32 = i + 1;
    } while (next < n);
    _irf[rd] = acc;
}
''')
    mlir = tmp_path / 'input.mlir'
    run([ROOT / 'tools/aps-frontend', 'mlir', cadl, '-o', mlir])
    result = subprocess.run([str(E2E), '-i', str(mlir), '--resource',
                             str(resource), '--export', 'cmt2', '-o',
                             str(tmp_path / 'output.mlir')], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'pipeline loop with iter_args/results is not supported' in result.stderr


def test_flattened_addresses_keep_partitioning(tmp_path, resource):
    declarations = ''.join(f'''#[partition_dim_array([0])]
#[partition_factor_array([4])]
#[partition_cyclic_array([1])]
static {name}: [u32; 64];
''' for name in ['a', 'b', 'c'])
    output = compile_cadl(tmp_path, declarations + '''#[opcode(7'b0001011)]
#[funct7(7'b0000000)]
rtype flat(rs1:u5, rs2:u5, rd:u5) {
    [[unroll(2)]]
    [[pipeline(1)]]
    [[II(1)]]
    with p:u32 = (0, next) do {
        let row:u32 = p / 4;
        let col:u32 = p % 4;
        c[p] = a[row * 4 + 1] + b[col * 4 + 2];
        let next:u32 = p + 1;
    } while (next < 16);
    _irf[rd] = 0;
}
''', resource, export='cmt2')
    text = output.read_text()
    for name in ['a', 'b', 'c']:
        for bank in range(4):
            assert f'@BankWrapper_{name}_{bank}(' in text
    assert 'partition pragma' not in (tmp_path / 'compile.log').read_text()
