#!/usr/bin/env python3
"""
Test script for the new APS dialect memory operations
"""

import circt
import circt.ir as ir
import circt.dialects.aps as aps
import circt.dialects.func as func

def test_aps_memory_ops():
    """Test APS dialect memory operations"""
    print("Testing APS dialect memory operations...")

    with ir.Context() as ctx:
        # Register CIRCT dialects
        circt.register_dialects(ctx)

        # Create a module
        with ir.Location.unknown():
            module = ir.Module.create()

            with ir.InsertionPoint(module.body):
                # Test MemDeclare
                element_type = ir.IntegerType.get_signless(32)
                memory_type = ir.MemRefType.get([ir.ShapedType.get_dynamic_size()], element_type)

                # Create a test function
                func_type = ir.FunctionType.get([], [])
                test_func = func.FuncOp("test_aps_memory", func_type)

                # Add entry block
                entry_block = test_func.add_entry_block()

                with ir.InsertionPoint(entry_block):
                    # Test MemDeclare operation
                    print("Testing aps.memdeclare...")
                    mem_declare = aps.MemDeclare(memory_type)
                    print("✅ aps.memdeclare created successfully")

                    # Test constants for addressing
                    addr_const = ir.IntegerAttr.get(ir.IndexType.get(), 42)
                    addr = aps.ConstantOp(ir.IndexType.get(), addr_const).result if hasattr(aps, 'ConstantOp') else None

                    # If APS doesn't have ConstantOp, use arith
                    if addr is None:
                        import circt.dialects.arith as arith
                        addr = arith.ConstantOp(ir.IndexType.get(), 42).result

                    # Test value constant
                    val_const = ir.IntegerAttr.get(element_type, 123)
                    import circt.dialects.arith as arith
                    val = arith.ConstantOp(element_type, 123).result

                    # Test MemStore operation
                    print("Testing aps.memstore...")
                    aps.MemStore(val, mem_declare.result, [addr])
                    print("✅ aps.memstore created successfully")

                    # Test MemLoad operation
                    print("Testing aps.memload...")
                    loaded_val = aps.MemLoad(element_type, mem_declare.result, [addr])
                    print("✅ aps.memload created successfully")

                    # Return from function
                    func.ReturnOp([])

            # Print the generated MLIR
            print("\n=== Generated MLIR ===")
            print(module)

            print("\n✅ All APS dialect operations work correctly!")
            return True

def test_mlir_converter_integration():
    """Test the MLIR converter with APS operations"""
    print("\nTesting MLIR converter integration...")

    # Simple CADL code with memory operations
    cadl_code = '''
    static mem_size: u32 = 1024;

    function test_mem() -> u32 {
        let addr: u32 = 10;
        let value: u32 = 42;
        _mem[addr] = value;
        return _mem[addr];
    }
    '''

    try:
        from cadl_frontend import parse_proc
        from cadl_frontend.mlir_converter import convert_cadl_to_mlir

        # Parse CADL
        print("Parsing CADL code...")
        ast = parse_proc(cadl_code, "test.cadl")
        print("✅ CADL parsed successfully")

        # Convert to MLIR
        print("Converting to MLIR...")
        mlir_module = convert_cadl_to_mlir(ast)
        print("✅ MLIR conversion successful")

        # Print the generated MLIR
        print("\n=== Generated MLIR from CADL ===")
        print(mlir_module)

        return True

    except Exception as e:
        print(f"❌ MLIR converter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing APS Dialect Memory Operations\n")

    # Test basic APS operations
    success1 = test_aps_memory_ops()

    # Test MLIR converter integration
    success2 = test_mlir_converter_integration()

    if success1 and success2:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")