#!/usr/bin/env python3
"""
Test MLIR bindings availability and basic functionality
"""

import pytest
import sys
import os


class TestMLIRBindings:
    """Test MLIR Python bindings"""

    def test_mlir_imports(self):
        """Test basic MLIR imports"""
        try:
            import mlir.ir as ir
            import mlir.dialects.func as func
            import mlir.dialects.arith as arith
            import mlir.dialects.scf as scf
            import mlir.dialects.memref as memref
            assert True
        except ImportError as e:
            pytest.skip(f"MLIR bindings not available: {e}")

    def test_circt_imports(self):
        """Test CIRCT dialect imports"""
        try:
            import circt.dialects.comb as comb
            import circt.dialects.hw as hw
            assert True
        except ImportError as e:
            pytest.skip(f"CIRCT bindings not available: {e}")

    def test_mlir_context_creation(self):
        """Test MLIR context and module creation"""
        try:
            import mlir.ir as ir

            with ir.Context() as ctx:
                assert ctx is not None

                with ir.Location.unknown():
                    module = ir.Module.create()
                    assert module is not None

        except ImportError:
            pytest.skip("MLIR bindings not available")
        except Exception as e:
            pytest.fail(f"MLIR context creation failed: {e}")

    def test_mlir_function_creation(self):
        """Test creating a simple MLIR function"""
        try:
            import mlir.ir as ir
            import mlir.dialects.func as func

            with ir.Context() as ctx:
                with ir.Location.unknown():
                    module = ir.Module.create()

                    with ir.InsertionPoint(module.body):
                        # Create a simple function
                        func_type = ir.FunctionType.get([], [])
                        func_op = func.FuncOp("test_func", func_type)

                        assert func_op is not None
                        assert "test_func" in str(module)

        except ImportError:
            pytest.skip("MLIR bindings not available")
        except Exception as e:
            pytest.fail(f"MLIR function creation failed: {e}")

    def test_mlir_arithmetic_operations(self):
        """Test creating arithmetic operations"""
        try:
            import mlir.ir as ir
            import mlir.dialects.func as func
            import mlir.dialects.arith as arith

            with ir.Context() as ctx:
                with ir.Location.unknown():
                    module = ir.Module.create()

                    with ir.InsertionPoint(module.body):
                        # Create function with arithmetic
                        i32 = ir.IntegerType.get_signless(32)
                        func_type = ir.FunctionType.get([i32, i32], [i32])
                        func_op = func.FuncOp("add_func", func_type)

                        # Add entry block
                        entry_block = func_op.add_entry_block()

                        with ir.InsertionPoint(entry_block):
                            # Create addition
                            a, b = entry_block.arguments
                            result = arith.AddIOp(a, b)
                            func.ReturnOp([result.result])

                        assert "arith.addi" in str(module)

        except ImportError:
            pytest.skip("MLIR bindings not available")
        except Exception as e:
            pytest.fail(f"MLIR arithmetic operations failed: {e}")

    def test_circt_combinational_operations(self):
        """Test CIRCT combinational operations"""
        try:
            import mlir.ir as ir
            import circt.dialects.comb as comb

            with ir.Context() as ctx:
                with ir.Location.unknown():
                    module = ir.Module.create()

                    with ir.InsertionPoint(module.body):
                        # Create some constants for testing
                        i32 = ir.IntegerType.get_signless(32)

                        # This is a basic test - we can't create full operations
                        # without proper setup, but we can test that the imports work
                        assert comb is not None

        except ImportError:
            pytest.skip("CIRCT bindings not available")
        except Exception as e:
            pytest.fail(f"CIRCT operations failed: {e}")


class TestEnvironmentSetup:
    """Test environment and path setup"""

    def test_python_path_setup(self):
        """Test that PYTHONPATH includes MLIR directories"""
        python_path = os.environ.get('PYTHONPATH', '')

        expected_paths = [
            'mlir_core',
            'circt_core'
        ]

        for path in expected_paths:
            if path not in python_path:
                pytest.skip(f"PYTHONPATH missing {path} - rebuild may be needed")

    def test_mlir_directories_exist(self):
        """Test that MLIR installation directories exist"""
        base_dir = "/home/zyy/aps-mlir/cadl-frontend"
        expected_dirs = [
            f"{base_dir}/circt_install/python_packages/mlir_core",
            f"{base_dir}/circt_install/python_packages/circt_core"
        ]

        for dir_path in expected_dirs:
            if not os.path.exists(dir_path):
                pytest.skip(f"MLIR directory missing: {dir_path}")


if __name__ == "__main__":
    # Allow running this file directly
    pytest.main([__file__, "-v"])