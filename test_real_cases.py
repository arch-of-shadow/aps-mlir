#!/usr/bin/env python3
"""
Test MLIR converter with real CADL examples
"""

import sys
import os
import traceback
from pathlib import Path

# Add cadl_frontend to path
sys.path.insert(0, os.path.dirname(__file__))

from cadl_frontend.parser import parse_proc
from cadl_frontend.mlir_converter import convert_cadl_to_mlir


def test_cadl_file(filepath: str) -> bool:
    """Test a single CADL file conversion to MLIR"""
    print(f"\n{'='*60}")
    print(f"Testing: {filepath}")
    print('='*60)

    try:
        # Read the file
        with open(filepath, 'r') as f:
            source = f.read()

        # Parse CADL to AST
        print("1. Parsing CADL to AST...")
        ast = parse_proc(source, filepath)

        # Print AST statistics
        print(f"   ✓ Parsed successfully!")
        print(f"   - Functions: {len(ast.functions)}")
        print(f"   - Flows: {len(ast.flows)}")
        print(f"   - Statics: {len(ast.statics)}")
        print(f"   - Regfiles: {len(ast.regfiles)}")

        # List the flows/functions
        if ast.flows:
            print("   - Flow names:", ', '.join(ast.flows.keys()))
        if ast.functions:
            print("   - Function names:", ', '.join(ast.functions.keys()))

        # Convert to MLIR
        print("\n2. Converting AST to MLIR...")
        mlir_module = convert_cadl_to_mlir(ast)

        print("   ✓ Conversion successful!")

        # Print MLIR output (first 500 chars)
        mlir_str = str(mlir_module)
        print("\n3. MLIR Output (preview):")
        print("-" * 40)
        if len(mlir_str) > 500:
            print(mlir_str[:500] + "...")
        else:
            print(mlir_str)

        # Count operations in MLIR
        op_count = mlir_str.count('"func.func"')
        print(f"\n   - Generated {op_count} function(s) in MLIR")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "--verbose" in sys.argv:
            print("\nFull traceback:")
            traceback.print_exc()
        return False


def test_zyy_cadl():
    """Test the complex zyy.cadl file"""
    print("\n" + "="*60)
    print("SPECIAL TEST: zyy.cadl (Complex real-world example)")
    print("="*60)

    filepath = "examples/zyy.cadl"

    try:
        with open(filepath, 'r') as f:
            source = f.read()

        # Count commented vs uncommented flows
        lines = source.split('\n')
        commented_rtypes = sum(1 for line in lines if line.strip().startswith('// rtype'))
        active_rtypes = sum(1 for line in lines if line.strip().startswith('rtype'))

        print(f"File statistics:")
        print(f"  - Total lines: {len(lines)}")
        print(f"  - Active rtype flows: {active_rtypes}")
        print(f"  - Commented rtype flows: {commented_rtypes}")

        # Parse only the active parts
        print("\nParsing active code...")
        ast = parse_proc(source, filepath)

        print(f"Successfully parsed:")
        for flow_name, flow in ast.flows.items():
            attrs = []
            if hasattr(flow, 'attrs') and flow.attrs:
                # Check if attrs is dict-like or has attributes
                try:
                    if hasattr(flow.attrs, 'get'):  # dict-like
                        if flow.attrs.get('opcode'):
                            attrs.append(f"opcode={flow.attrs['opcode']}")
                        if flow.attrs.get('funct7'):
                            attrs.append(f"funct7={flow.attrs['funct7']}")
                    elif hasattr(flow.attrs, 'opcode'):  # object with attributes
                        if flow.attrs.opcode:
                            attrs.append(f"opcode={flow.attrs.opcode}")
                        if hasattr(flow.attrs, 'funct7') and flow.attrs.funct7:
                            attrs.append(f"funct7={flow.attrs.funct7}")
                except:
                    pass  # Skip if attrs access fails
            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
            print(f"  - rtype {flow_name}{attr_str}")

        # Try MLIR conversion
        print("\nAttempting MLIR conversion...")
        mlir_module = convert_cadl_to_mlir(ast)
        print("✓ MLIR conversion successful!")

        # Show MLIR functions
        mlir_str = str(mlir_module)
        func_names = []
        for line in mlir_str.split('\n'):
            if 'sym_name = "' in line:
                start = line.index('sym_name = "') + len('sym_name = "')
                end = line.index('"', start)
                func_names.append(line[start:end])

        if func_names:
            print(f"Generated MLIR functions: {', '.join(func_names)}")

        return True

    except Exception as e:
        print(f"\n❌ Error with zyy.cadl: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def main():
    """Run tests on real CADL files"""
    print("CADL to MLIR Converter - Real Case Testing")
    print("=" * 60)

    # Test files
    test_files = [
        "examples/simple.cadl",
        "examples/zyy.cadl",
    ]

    # Check which files exist
    existing_files = []
    for filepath in test_files:
        if os.path.exists(filepath):
            existing_files.append(filepath)
            print(f"✓ Found: {filepath}")
        else:
            print(f"✗ Missing: {filepath}")

    if not existing_files:
        print("\n❌ No test files found!")
        return 1

    # Run tests
    results = {}

    # First test simple files
    for filepath in existing_files:
        if filepath != "examples/zyy.cadl":
            results[filepath] = test_cadl_file(filepath)

    # Special test for zyy.cadl
    if "examples/zyy.cadl" in existing_files:
        results["examples/zyy.cadl"] = test_zyy_cadl()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for filepath, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {filepath}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())