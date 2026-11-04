from cadl_frontend import parse_proc
from cadl_frontend.to_mlir import memory as mlir_memory
from cadl_frontend import cadl_ast


def first_flow(source: str):
    return next(iter(parse_proc(source).flows.values()))


def test_flow_uses_memory_for_mem_read():
    flow = first_flow("""
    rtype test(addr: u32, rd: u5) {
        let value: u32 = _mem[addr];
        _irf[rd] = value;
    }
    """)
    assert mlir_memory.flow_uses_memory(flow)
    assert mlir_memory.stmt_uses_memory(flow.body[0])
    assert mlir_memory.expr_uses_memory(flow.body[0].rhs)


def test_flow_uses_memory_for_mem_write():
    flow = first_flow("""
    rtype test(addr: u32, value: u32) {
        _mem[addr] = value;
    }
    """)
    assert isinstance(flow.body[0], cadl_ast.AssignStmt)
    assert mlir_memory.flow_uses_memory(flow)
    assert mlir_memory.stmt_uses_memory(flow.body[0])


def test_flow_without_mem_does_not_use_memory():
    flow = first_flow("""
    rtype test(rs1: u5, rs2: u5, rd: u5) {
        let a: u32 = _irf[rs1];
        let b: u32 = _irf[rs2];
        _irf[rd] = a + b;
    }
    """)
    assert not mlir_memory.flow_uses_memory(flow)
    assert not mlir_memory.stmt_list_uses_memory(flow.body)


def test_loop_body_mem_read_is_detected():
    flow = first_flow("""
    rtype test(addr: u32, rd: u5) {
        let i0: u32 = 0;
        let sum0: u32 = 0;
        with
            i: u32 = (i0, i + 1)
            sum: u32 = (sum0, sum + _mem[addr])
        do {
            let i_: u32 = i + 1;
            let sum_: u32 = sum + _mem[addr];
        } while (i < 4);
        _irf[rd] = sum;
    }
    """)
    loop = flow.body[2]
    assert isinstance(loop, cadl_ast.DoWhileStmt)
    assert mlir_memory.flow_uses_memory(flow)
    assert mlir_memory.stmt_uses_memory(loop)


def test_mem_in_call_arguments_is_detected():
    flow = first_flow("""
    rtype test(addr: u32) {
        foo(_mem[addr]);
    }
    """)
    assert mlir_memory.flow_uses_memory(flow)
