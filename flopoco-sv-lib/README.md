# Flopoco SystemVerilog Library

Library containing SystemVerilog Wrappers and Helper for Flopoco Floating Point IPs.

Verilog versions generated from Flopoco VHDL IPs can be found in the```flopoco``` directory.

## Available Modules

### Formats
- 18bit and 34bit Flopoco Floating Point
- 16bit IEEE-754 Half Precision Floating Point
- 32bit IEEE-754 Single Precision Floating Point

### Flopoco Wrappers
Can be found in ```rtl/wrappers```

- **FP2IEEE** Flopoco Floating Point to IEEE-754
- **FP2INT**  Flopoco Floating Point to Signed Integer
- **FPComp**  Flopoco Floating Point Comparator
- **FPDiv**   Flopoco Floating Point Divider
- **FPLog**   Flopoco Floating Point Logarithm
- **FPSqrt**  Flopoco Floating Point Square Root
- **IEEE2FP** IEEE-754 to Flopoco Floating Point Converter
- **IEEEExp** IEEE-754 Exponential Function
- **IEEEFMA** IEEE-754 Fused Multiply-Add
- **INT2FP**  Signed Integer to Flopoco Floating Point

### Helpers using Flopoco Wrappers
Can be found in ```rtl/```

- **IEEE2INT** IEEE-754 to Signed Integer Converter
- **IEEEComp** IEEE-754 Comparator
- **IEEEDiv**  IEEE-754 Divider
- **IEEELog**  IEEE-754 Logarithm
- **IEEESqrt** IEEE-754 Square Root
- **INT2IEEE** Signed Integer to Flopoco Floating Point

## License
All code in this repository should have a permissive license. Hardware is licensed under Solderpad Hardware License 0.51 (see [`LICENSE-SHL`](LICENSE-SHL))

## Copyright

© Tobias Senti 2025
