# Berkeley HardFloat (Verilog Release 1)

Unmodified `source/` and `COPYING.txt` from the official Release 1 archive:

- URL: https://www.jhauser.us/arithmetic/HardFloat-1.zip
- Archive SHA-256: `6b3757c9fbfa2230c6a2b84605e39372cb589dd7500e979c4f0b8ecc8a03b14b`
- Upstream documentation: https://www.jhauser.us/arithmetic/HardFloat-1/doc/HardFloat-Verilog.html
- License: BSD-style, see `COPYING.txt` and source headers.

AQUAS selects the **RISCV** specialization: canonical quiet NaNs, RNE,
tininess detected after rounding, and gradual underflow. Other specialization
files are retained as upstream shipped them, but are not compiled.

This is the official **Verilog** release, not the Chisel HardFloat commit in
Chipyard. ISAX IP generation does not require Chipyard or Scala/SBT.

Reproduce the vendor import by verifying the archive hash, then copying only
`HardFloat-1/source/` and `HardFloat-1/COPYING.txt`. `SOURCE.sha256` records every
vendored file relative to this directory; verify with `sha256sum -c SOURCE.sha256`.
