"""
Test cases for declarations

Tests static variables, regfiles, and their attributes.
"""

import pytest
from cadl_frontend import parse_proc
from cadl_frontend import cadl_ast


class TestRegfileDeclarations:
    """Test regfile declaration parsing"""

    def test_regfile_definition(self):
        """Test regfile definition parsing"""
        source = "regfile rf(32, 16);"
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.regfiles) == 1

        regfile = list(ast.regfiles.values())[0]
        assert regfile.name == "rf"
        assert regfile.width == 32
        assert regfile.depth == 16

    def test_multiple_regfiles(self):
        """Test multiple regfile declarations"""
        source = """
        regfile rf1(32, 16);
        regfile rf2(64, 32);
        """
        ast = parse_proc(source)
        assert len(ast.regfiles) == 2


class TestStaticDeclarations:
    """Test static variable declarations"""

    def test_simple_static(self):
        """Test simple static declaration"""
        source = "static counter: u32;"
        ast = parse_proc(source)
        assert len(ast.statics) == 1

        static = list(ast.statics.values())[0]
        assert static.id == "counter"
        assert isinstance(static.ty, cadl_ast.DataType_Single)

    def test_static_with_initialization(self):
        """Test static with initialization"""
        source = "static counter: u32 = 42;"
        ast = parse_proc(source)
        static = list(ast.statics.values())[0]
        assert static.expr is not None
        assert isinstance(static.expr, cadl_ast.LitExpr)

    def test_static_array(self):
        """Test static array declaration"""
        source = "static buffer: [i32; 1024];"
        ast = parse_proc(source)
        static = list(ast.statics.values())[0]
        assert isinstance(static.ty, cadl_ast.DataType_Array)

    def test_static_array_with_aggregate(self):
        """Test static array with aggregate initialization"""
        source = "static arr: [u32; 3] = {1, 2, 3};"
        ast = parse_proc(source)
        static = list(ast.statics.values())[0]
        assert isinstance(static.expr, cadl_ast.AggregateExpr)
        assert len(static.expr.elements) == 3


class TestStaticAttributes:
    """Test attribute parsing on static declarations"""

    def test_static_with_single_attribute(self):
        """Test static with single attribute"""
        source = """
        #[impl("1rw")]
        static buffer: [i32; 1024];
        """
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.statics) == 1

        buffer = ast.statics["buffer"]
        assert buffer.attrs is not None
        assert "impl" in buffer.attrs
        assert isinstance(buffer.attrs["impl"], cadl_ast.StringLitExpr)
        assert buffer.attrs["impl"].value == "1rw"

    def test_static_with_multiple_attributes(self):
        """Test static with multiple attributes"""
        source = """
        #[impl("2rw")]
        #[partition("cyclic")]
        static scratch: [i32; 512];
        """
        ast = parse_proc(source)
        assert ast is not None

        scratch = ast.statics["scratch"]
        assert len(scratch.attrs) == 2
        assert "impl" in scratch.attrs
        assert "partition" in scratch.attrs

        assert isinstance(scratch.attrs["impl"], cadl_ast.StringLitExpr)
        assert scratch.attrs["impl"].value == "2rw"

        assert isinstance(scratch.attrs["partition"], cadl_ast.StringLitExpr)
        assert scratch.attrs["partition"].value == "cyclic"

    def test_static_with_attribute_and_init(self):
        """Test static with attribute and initialization"""
        source = """
        #[impl("1rw")]
        static counter: i32 = 42;
        """
        ast = parse_proc(source)
        assert ast is not None

        counter = ast.statics["counter"]
        assert "impl" in counter.attrs
        assert counter.expr is not None
        assert isinstance(counter.expr, cadl_ast.LitExpr)

    def test_static_without_attributes(self):
        """Test static without attributes (regression test)"""
        source = """
        static normal: [i32; 256];
        """
        ast = parse_proc(source)
        assert ast is not None

        normal = ast.statics["normal"]
        assert len(normal.attrs) == 0

    def test_static_attribute_with_integer_value(self):
        """Test static with integer attribute value"""
        source = """
        #[factor(4)]
        static data: [i32; 1024];
        """
        ast = parse_proc(source)
        assert ast is not None

        data = ast.statics["data"]
        assert "factor" in data.attrs
        assert isinstance(data.attrs["factor"], cadl_ast.LitExpr)

    def test_mixed_statics_with_and_without_attributes(self):
        """Test multiple statics, some with attributes, some without"""
        source = """
        #[impl("1rw")]
        static buffer1: [i32; 512];

        static buffer2: [i32; 256];

        #[impl("2rw")]
        #[partition("block")]
        static buffer3: [i32; 128];
        """
        ast = parse_proc(source)
        assert ast is not None
        assert len(ast.statics) == 3

        buffer1 = ast.statics["buffer1"]
        assert len(buffer1.attrs) == 1
        assert "impl" in buffer1.attrs

        buffer2 = ast.statics["buffer2"]
        assert len(buffer2.attrs) == 0

        buffer3 = ast.statics["buffer3"]
        assert len(buffer3.attrs) == 2
        assert "impl" in buffer3.attrs
        assert "partition" in buffer3.attrs


class TestCSRRegisterDeclarations:
    """Test custom CSR register declaration parsing"""

    def test_csr_register_with_address(self):
        """Test CSR register declaration with address attribute"""
        source = """
        #[csr_address(0x801)]
        register cfg: u32;
        """
        ast = parse_proc(source)

        assert len(ast.registers) == 1
        assert len(ast.csrs) == 1
        csr = ast.csrs["cfg"]
        assert csr.name == "cfg"
        assert csr.is_csr
        assert isinstance(csr.ty, cadl_ast.DataType_Single)
        assert "csr_address" in csr.attrs
        assert csr.address == 0x801

    def test_multiple_csr_registers(self):
        """Test multiple CSR register declarations"""
        source = """
        #[csr_address(0x801)]
        register cfg: u32;

        #[csr_address(0x802)]
        register status: u32;
        """
        ast = parse_proc(source)

        assert list(ast.registers.keys()) == ["cfg", "status"]
        assert list(ast.csrs.keys()) == ["cfg", "status"]
        assert ast.csrs["cfg"].address == 0x801
        assert ast.csrs["status"].address == 0x802

    def test_csr_access_uses_special_csr_index(self):
        """Test _csr[name] parses as an indexed CSR special expression"""
        source = """
        #[csr_address(0x801)]
        register cfg: u32;

        rtype test(rd: u5) {
            let value: u32 = _csr[cfg];
            _irf[rd] = value;
        }
        """
        ast = parse_proc(source)
        flow = ast.flows["test"]
        assign = flow.body[0]

        assert isinstance(assign.rhs, cadl_ast.IndexExpr)
        assert isinstance(assign.rhs.expr, cadl_ast.IdentExpr)
        assert assign.rhs.expr.name == "_csr"
        assert isinstance(assign.rhs.indices[0], cadl_ast.IdentExpr)
        assert assign.rhs.indices[0].name == "cfg"

    def test_register_without_csr_is_not_implemented(self):
        """Test non-CSR register declarations are accepted by syntax but unsupported"""
        source = """
        register state: u32;
        """
        with pytest.raises(Exception, match="Only #\\[csr_address"):
            parse_proc(source)

    def test_register_type_first_is_rejected(self):
        """Test register: type name syntax is intentionally rejected"""
        source = """
        #[csr_address(0x801)]
        register: u32 cfg;
        """
        with pytest.raises(Exception):
            parse_proc(source)


class TestDeclarationCombinations:
    """Test combinations of declarations"""

    def test_mixed_declarations(self):
        """Test mixing regfile and static declarations"""
        source = """
        regfile rf(32, 16);
        static counter: u32 = 0;
        static buffer: [i32; 256];
        """
        ast = parse_proc(source)
        assert len(ast.regfiles) == 1
        assert len(ast.statics) == 2

    def test_declarations_with_flows(self):
        """Test declarations alongside flow definitions"""
        source = """
        static data: [u32; 100];

        rtype process() {
            data[0] = 42;
        }
        """
        ast = parse_proc(source)
        assert len(ast.statics) == 1
        assert len(ast.flows) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
