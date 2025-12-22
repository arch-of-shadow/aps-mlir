#!/usr/bin/env python3

import os

content = """
# Copyright 2025 Tobias Senti
# Solderpad Hardware License, Version 0.51, see LICENSE for details.
# SPDX-License-Identifier: SHL-0.51

package:
    name: flopoco-sv-lib
    authors:
        - "Tobias Senti <git@tsenti.li>"

sources:
"""

mapping = {
    "Fix2FP": "INT to Flopoco Floating Point conversion",
    "FPComp": "Flopoco Floating Point comparison",
    "FPDiv": "Flopoco Floating Point division",
    "FPSqrt": "Flopoco Floating Point square root",
    "FPLog": "Flopoco Floating Point logarithm",
    "FP2Fix": "Flopoco Floating Point to INT conversion",
    "FP2IEEE": "Flopoco Floating Point to IEEE 754 conversion",
    "IEEE2FP": "IEEE 754 to Flopoco Floating Point conversion",
    "IEEEExp": "IEEE 754 exponentiation",
    "IEEEFMA": "IEEE 754 fused multiply-add",
}

for dir in os.listdir("flopoco"):
    if not os.path.isdir(os.path.join("flopoco", dir)):
        continue
    content += f"    # {mapping[dir]}\n"
    for file in os.listdir(os.path.join("flopoco", dir)):
        content += f"    - flopoco/{dir}/{file}\n"
    content += "\n"

content += """    # Wrappers
    # Operations
    - rtl/wrappers/FPComp.sv
    - rtl/wrappers/FPDiv.sv
    - rtl/wrappers/FPLog.sv
    - rtl/wrappers/FPSqrt.sv
    - rtl/wrappers/IEEEExp.sv
    - rtl/wrappers/IEEEFMA.sv

    # Flopoco FP to/from IEEE 754
    - rtl/wrappers/IEEE2FP.sv
    - rtl/wrappers/FP2IEEE.sv

    # Flopoco FP to/from INT
    - rtl/wrappers/INT2FP.sv
    - rtl/wrappers/FP2INT.sv

    # Conversion between INT to/from IEEE 754
    - rtl/INT2IEEE.sv
    - rtl/IEEE2INT.sv

    # IEEE 754 Comparator
    - rtl/IEEEComp.sv

    # IEEE 754 Division
    - rtl/IEEEDiv.sv

    # IEEE 754 Logarithm
    - rtl/IEEELog.sv

    # IEEE 754 Square Root
    - rtl/IEEESqrt.sv
"""

print(content)
with open("Bender.yml", "w") as f:
    f.write(content)
