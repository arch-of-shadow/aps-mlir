#!/usr/bin/env python3
"""
Test to verify that _mem and _irf special identifiers work correctly
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend.ast import *


class TestMemIrf:
    """Test _mem and _irf special identifier parsing"""

    def test_irf_read_access(self):
        """Test reading from integer register file"""
        source = """
        rtype irf_read(rs1: u5) {
            let r1: u32 = _irf[rs1];
            return (r1);
        }
        """
        
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.flows) == 1
        
        flow = list(ast.flows.values())[0]
        assert flow.name == "irf_read"
        
        # Check the assignment statement
        assign_stmt = flow.body[0]
        assert isinstance(assign_stmt, AssignStmt)
        assert assign_stmt.is_let == True
        
        # RHS should be index expression: _irf[rs1]
        assert isinstance(assign_stmt.rhs, IndexExpr)
        
        # Base expression should be identifier "_irf"
        assert isinstance(assign_stmt.rhs.expr, IdentExpr)
        assert assign_stmt.rhs.expr.name == "_irf"
        
        # Index should be identifier "rs1"
        assert len(assign_stmt.rhs.indices) == 1
        assert isinstance(assign_stmt.rhs.indices[0], IdentExpr)
        assert assign_stmt.rhs.indices[0].name == "rs1"

    def test_irf_write_access(self):
        """Test writing to integer register file"""
        source = """
        rtype irf_write(rs1: u5, rs2: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2];
            _irf[rd] = r1 + r2;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have 2 let assignments + 1 regular assignment
        assert len(flow.body) == 3
        
        # Check the write assignment (third statement)
        write_stmt = flow.body[2]
        assert isinstance(write_stmt, AssignStmt)
        assert write_stmt.is_let == False
        
        # LHS should be index expression: _irf[rd]
        assert isinstance(write_stmt.lhs, IndexExpr)
        assert isinstance(write_stmt.lhs.expr, IdentExpr)
        assert write_stmt.lhs.expr.name == "_irf"
        
        # RHS should be addition
        assert isinstance(write_stmt.rhs, BinaryExpr)
        assert write_stmt.rhs.op == BinaryOp.ADD

    def test_memory_read_access(self):
        """Test reading from memory"""
        source = """
        rtype mem_read(addr: u32) {
            let data: u32 = _mem[addr];
            return (data);
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check the assignment statement
        assign_stmt = flow.body[0]
        assert isinstance(assign_stmt, AssignStmt)
        
        # RHS should be index expression: _mem[addr]
        assert isinstance(assign_stmt.rhs, IndexExpr)
        assert isinstance(assign_stmt.rhs.expr, IdentExpr)
        assert assign_stmt.rhs.expr.name == "_mem"
        
        # Index should be "addr"
        assert len(assign_stmt.rhs.indices) == 1
        assert isinstance(assign_stmt.rhs.indices[0], IdentExpr)
        assert assign_stmt.rhs.indices[0].name == "addr"

    def test_memory_write_access(self):
        """Test writing to memory"""
        source = """
        rtype mem_write(addr: u32, value: u32) {
            _mem[addr] = value;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check the assignment statement
        assign_stmt = flow.body[0]
        assert isinstance(assign_stmt, AssignStmt)
        assert assign_stmt.is_let == False
        
        # LHS should be memory index expression
        assert isinstance(assign_stmt.lhs, IndexExpr)
        assert isinstance(assign_stmt.lhs.expr, IdentExpr)
        assert assign_stmt.lhs.expr.name == "_mem"
        
        # RHS should be identifier
        assert isinstance(assign_stmt.rhs, IdentExpr)
        assert assign_stmt.rhs.name == "value"

    def test_memory_with_address_calculation(self):
        """Test memory access with address calculation"""
        source = """
        rtype mem_calc_addr(base: u32, offset: u32) {
            let addr: u32 = base + offset;
            let data1: u32 = _mem[addr];
            let data2: u32 = _mem[base + 4];
            _mem[addr + 8] = data1 + data2;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have 3 assignments + 1 memory write
        assert len(flow.body) == 4
        
        # Check memory read with calculated address: _mem[base + 4]
        read_stmt = flow.body[2]
        assert isinstance(read_stmt.rhs, IndexExpr)
        assert isinstance(read_stmt.rhs.expr, IdentExpr)
        assert read_stmt.rhs.expr.name == "_mem"
        
        # Index should be binary expression: base + 4
        index_expr = read_stmt.rhs.indices[0]
        assert isinstance(index_expr, BinaryExpr)
        assert index_expr.op == BinaryOp.ADD
        
        # Check memory write with calculated address: _mem[addr + 8]
        write_stmt = flow.body[3]
        assert isinstance(write_stmt.lhs, IndexExpr)
        lhs_index = write_stmt.lhs.indices[0]
        assert isinstance(lhs_index, BinaryExpr)
        assert lhs_index.op == BinaryOp.ADD

    def test_irf_with_multiple_registers(self):
        """Test accessing multiple registers in one flow"""
        source = """
        rtype multiple_regs(rs1: u5, rs2: u5, rs3: u5, rd: u5) {
            let r1: u32 = _irf[rs1];
            let r2: u32 = _irf[rs2]; 
            let r3: u32 = _irf[rs3];
            let sum: u32 = r1 + r2 + r3;
            _irf[rd] = sum;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have 4 let assignments + 1 irf write
        assert len(flow.body) == 5
        
        # Check all _irf reads are parsed correctly
        for i in range(3):  # First 3 statements are _irf reads
            stmt = flow.body[i]
            assert isinstance(stmt.rhs, IndexExpr)
            assert stmt.rhs.expr.name == "_irf"
        
        # Check _irf write
        write_stmt = flow.body[4]
        assert isinstance(write_stmt.lhs, IndexExpr)
        assert write_stmt.lhs.expr.name == "_irf"

    def test_mixed_mem_irf_operations(self):
        """Test mixing memory and register file operations"""
        source = """
        rtype mixed_ops(rs1: u5, addr: u32, rd: u5) {
            let reg_val: u32 = _irf[rs1];
            let mem_val: u32 = _mem[addr];
            let combined: u32 = reg_val + mem_val;
            _irf[rd] = combined;
            _mem[addr + 4] = combined;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Should have 3 let assignments + 2 writes
        assert len(flow.body) == 5
        
        # Check first read is from _irf
        irf_read = flow.body[0]
        assert isinstance(irf_read.rhs, IndexExpr)
        assert irf_read.rhs.expr.name == "_irf"
        
        # Check second read is from _mem
        mem_read = flow.body[1]
        assert isinstance(mem_read.rhs, IndexExpr)
        assert mem_read.rhs.expr.name == "_mem"
        
        # Check _irf write
        irf_write = flow.body[3]
        assert isinstance(irf_write.lhs, IndexExpr)
        assert irf_write.lhs.expr.name == "_irf"
        
        # Check _mem write
        mem_write = flow.body[4]
        assert isinstance(mem_write.lhs, IndexExpr)
        assert mem_write.lhs.expr.name == "_mem"

    def test_complex_expression_with_mem_irf(self):
        """Test complex expressions involving _mem and _irf"""
        source = """
        rtype complex_expr(rs1: u5, base_addr: u32, rd: u5) {
            let reg1: u32 = _irf[rs1];
            let addr: u32 = base_addr + reg1;
            let mem_data: u32 = _mem[addr];
            let result: u32 = if mem_data > reg1 {_mem[addr + 4]} else {_irf[rs1]};
            _irf[rd] = result;
        }
        """
        
        ast = parse_proc(source)
        flow = list(ast.flows.values())[0]
        
        # Check the if expression in the assignment
        result_assign = flow.body[3]
        assert isinstance(result_assign.rhs, IfExpr)
        
        # Then branch should be _mem access
        then_branch = result_assign.rhs.then_branch
        assert isinstance(then_branch, IndexExpr)
        assert then_branch.expr.name == "_mem"
        
        # Else branch should be _irf access
        else_branch = result_assign.rhs.else_branch
        assert isinstance(else_branch, IndexExpr)
        assert else_branch.expr.name == "_irf"


if __name__ == "__main__":
    pytest.main([__file__])