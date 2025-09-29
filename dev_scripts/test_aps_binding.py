#!/usr/bin/env python3

def test_aps_dialect():
    try:
        import circt
        print('CIRCT imported successfully!')

        # Try to import APS dialect directly
        try:
            from circt.dialects import func, arith, memref
            from circt.dialects import aps
            print('APS dialect Python bindings are working!')
            print(f'APS dialect module: {aps}')

            # Try to access the _Dialect class
            if hasattr(aps, '_Dialect'):
                print(f'APS _Dialect class: {aps._Dialect}')
            else:
                print('No _Dialect class found in aps module')

            return True
        except ImportError as e:
            print(f'Error importing APS dialect: {e}')
            return False

    except ImportError as e:
        print(f'Error importing CIRCT: {e}')
        return False

if __name__ == '__main__':
    success = test_aps_dialect()
    if success:
        print('SUCCESS: APS dialect Python bindings are working!')
    else:
        print('FAILED: APS dialect Python bindings are not working')