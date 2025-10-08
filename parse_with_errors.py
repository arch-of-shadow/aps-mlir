#!/usr/bin/env python3
"""Parse CADL files with clean error output"""

import sys
from cadl_frontend import parse_proc
from cadl_frontend.parser import CADLParseError

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_with_errors.py <file.cadl> [--summary]")
        sys.exit(1)
    
    filename = sys.argv[1]
    summary_mode = len(sys.argv) > 2 and sys.argv[2] == '--summary'
    
    try:
        with open(filename, 'r') as f:
            source = f.read()
        
        ast = parse_proc(source, filename)
        
        if summary_mode:
            print(ast)
        else:
            print(ast.pretty_print())
        
    except CADLParseError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()