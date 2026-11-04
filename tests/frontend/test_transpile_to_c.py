from pathlib import Path

import pytest

from cadl_frontend.to_c import CTranspiler


def transpile_text(tmp_path: Path, source: str) -> str:
    cadl_file = tmp_path / "input.cadl"
    cadl_file.write_text(source)
    transpiler = CTranspiler()
    return transpiler.transpile(cadl_file)


def test_csr_reads_become_parameters_with_bindings(tmp_path: Path):
    source = """
#[csr_address(0x801)]
register gain: u32;
#[csr_address(0x802)]
register bias: u32;
rtype adjust(rs1: u5, rd: u5) {
    let x: u32 = _irf[rs1];
    let g: u32 = _csr[gain];
    _irf[rd] = x * g + _csr[bias] + _csr[gain];
}
rtype plain(rs1: u5, rd: u5) {
    _irf[rd] = _irf[rs1] + 1;
}
"""
    path = tmp_path / "csr.cadl"
    path.write_text(source)
    transpiler = CTranspiler()
    output = transpiler.transpile(path)
    assert "uint32_t adjust(uint32_t rs1_value, uint32_t rs2_value, uint32_t csr_gain_value, uint32_t csr_bias_value)" in output
    assert "uint32_t g = csr_gain_value;" in output
    assert "_csr" not in output
    assert transpiler.csr_inputs == {
        "adjust": [
            {"name": "gain", "address": 0x801, "arg_index": 2},
            {"name": "bias", "address": 0x802, "arg_index": 3},
        ],
        "plain": [],
    }


@pytest.mark.parametrize("access, message", [
    ("_csr[missing]", "Undefined CSR"),
    ("_csr[0]", "declared register name"),
])
def test_invalid_csr_access(tmp_path: Path, access: str, message: str):
    with pytest.raises(ValueError, match=message):
        transpile_text(tmp_path, f"rtype bad(rd: u5) {{ _irf[rd] = {access}; }}")


@pytest.mark.parametrize("address, ty, message", [
    ("0x1000", "u32", "12-bit address"),
    ("0xc01", "u32", "read-only address"),
    ("0x801", "u64", "RV32 XLEN"),
])
def test_invalid_csr_declaration_for_megg(tmp_path: Path, address: str, ty: str, message: str):
    with pytest.raises(ValueError, match=message):
        transpile_text(tmp_path, f"""
#[csr_address({address})]
register cfg: {ty};
rtype bad(rd: u5) {{ _irf[rd] = _csr[cfg]; }}
""")


def test_csr_writes_rejected_in_matching_pattern(tmp_path: Path):
    with pytest.raises(ValueError, match="CSR writes"):
        transpile_text(tmp_path, """
#[csr_address(0x801)]
register cfg: u32;
rtype bad(rs1: u5) { _csr[cfg] = _irf[rs1]; }
""")


def test_register_read_converts_to_value_parameter(tmp_path: Path):
    cadl_source = """
rtype add1(rs1: u5, rd: u5) {
    let x: u16 = _irf[rs1];
    _irf[rd] = x + 1;
}
"""
    output = transpile_text(tmp_path, cadl_source)

    assert "uint16_t add1(uint16_t rs1_value, uint32_t rs2_value)" in output
    assert "_irf" not in output
    assert "return rd_result;" in output


def test_burst_access_uses_register_pointer(tmp_path: Path):
    cadl_source = """
static buf_in: [u32; 4];
static buf_out: [u32; 4];

rtype burst_demo(rs1: u5, rs2: u5, rd: u5) {
  let addr: u32 = _irf[rs1];
  buf_in[0 +: ] = _mem[addr +: 4];

  let v0: u32 = buf_in[0];
  buf_out[0] = v0 + 1;

  let out_addr: u32 = _irf[rs2];
  _mem[out_addr] = buf_out[0];
  _irf[rd] = buf_out[0];
}
"""

    output = transpile_text(tmp_path, cadl_source)

    assert "uint32_t burst_demo(uint32_t *rs1, uint32_t *rs2)" in output
    assert "uint32_t *rs1" in output
    assert "uint32_t *rs2" in output
    assert "rs1[0]" in output
    assert "rs2[0]" in output
    assert "_mem" not in output


def test_mem_access_rewritten_to_pointer(tmp_path: Path):
    cadl_source = """
rtype mem_demo(rs2: u5, rd: u5) {
  let base: u32 = _irf[rs2];
  let v0: u32 = _mem[base];
  let v1: u32 = _mem[base + 4];
  _irf[rd] = v0 + v1;
}
"""

    output = transpile_text(tmp_path, cadl_source)

    assert "uint32_t mem_demo(uint32_t rs1_value, uint32_t *rs2)" in output
    assert "_mem" not in output
    assert "rs2[0]" in output
    assert "rs2[1]" in output


def test_mem_store_rewritten_to_pointer(tmp_path: Path):
    cadl_source = """
rtype store_demo(rs2: u5) {
  let base: u32 = _irf[rs2];
  _mem[base] = 7;
  _mem[base + 4] = 9;
}
"""

    output = transpile_text(tmp_path, cadl_source)

    assert "void store_demo(uint32_t rs1_value, uint32_t *rs2)" in output
    assert "_mem" not in output
    assert "rs2[0] = 7;" in output
    assert "rs2[1] = 9;" in output


def test_static_without_burst_declared_locally(tmp_path: Path):
    cadl_source = """
static accum: [u32; 4];
static bias: u32;

rtype static_demo(rd: u5) {
  accum[1] = 7;
  let tmp: u32 = bias + accum[1];
  _irf[rd] = tmp;
}
"""

    output = transpile_text(tmp_path, cadl_source)

    assert "uint32_t static_demo(uint32_t rs1_value, uint32_t rs2_value)" in output
    assert "*accum" not in output
    assert "uint32_t accum[4];" in output
    assert "uint32_t bias = 0;" in output


def test_local_scalar_static_not_dereferenced(tmp_path: Path):
    cadl_source = """
static buf: [u32; 4];
static acc: u32;

rtype scalar_demo(rs1: u5) {
  buf[0 +: ] = _mem[_irf[rs1] +: 4];
  with i: u32 = (0, i_) do {
    acc = 0;
    acc = acc + buf[i];
    let i_: u32 = i + 1;
  } while (i_ < 4);
}
"""

    output = transpile_text(tmp_path, cadl_source)

    assert "uint32_t acc = 0;" in output
    assert "acc = 0;" in output
    assert "acc = (acc +" in output
    assert "*acc" not in output


def test_loop_lowers_to_for_when_possible(tmp_path: Path):
    cadl_source = """
rtype sum_loop(rs1: u5) {
    let base: u32 = _irf[rs1];
    with i: u32 = (0, i_) do {
        let value: u32 = base + i;
        let i_: u32 = i + 1;
    } while (i_ < 4);
}
"""
    output = transpile_text(tmp_path, cadl_source)

    assert "for (" in output
    assert "while (1)" not in output


def test_multiple_return_values_create_struct(tmp_path: Path):
    cadl_source = """
rtype pair(rs1: u5, rs2: u5) {
    let x: u32 = _irf[rs1];
    let y: u32 = _irf[rs2];
    _irf[rs1] = x;
    _irf[rs2] = y;
}
"""
    output = transpile_text(tmp_path, cadl_source)

    assert "typedef struct" in output
    assert "pair_result_t" in output
    assert "return (pair_result_t){" in output
