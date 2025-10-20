from pathlib import Path

import pytest

from cadl_frontend.transpile_to_c import CTranspiler


def transpile_text(tmp_path: Path, source: str) -> str:
    cadl_file = tmp_path / "input.cadl"
    cadl_file.write_text(source)
    transpiler = CTranspiler()
    return transpiler.transpile(cadl_file)


def test_register_read_converts_to_value_parameter(tmp_path: Path):
    cadl_source = """
rtype add1(rs1: u5, rd: u5) {
    let x: u16 = _irf[rs1];
    _irf[rd] = x + 1;
}
"""
    output = transpile_text(tmp_path, cadl_source)

    assert "uint16_t add1(uint16_t rs1_value)" in output
    assert "_irf" not in output
    assert "return rd_result;" in output


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
