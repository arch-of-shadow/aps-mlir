// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// IEEE 754 Floating Point comparator
module IEEEComp #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Dependent parameter, do **not** overwrite.
    parameter type ieee_t = logic [DataWidth-1:0]
) (
    /// Input operands
    input ieee_t operand_a_i,
    input ieee_t operand_b_i,

    /// Output result
    output logic a_lt_b_o,
    output logic a_eq_b_o,
    output logic a_gt_b_o,
    output logic a_le_b_o,
    output logic a_ge_b_o,
    output logic unordered_o
);
    // Flopoco Floating Point
    logic [DataWidth+1:0] flopoco_fp_a, flopoco_fp_b;

    // Convert IEEE 754 to Flopoco FP
    IEEE2FP #(
        .DataWidth( DataWidth )
    ) i_ieee2fp_a (
        .ieee_i( operand_a_i  ),
        .fp_o  ( flopoco_fp_a )
    );

    IEEE2FP #(
        .DataWidth( DataWidth )
    ) i_ieee2fp_b (
        .ieee_i( operand_b_i  ),
        .fp_o  ( flopoco_fp_b )
    );

    // Comparator
    FPComp #(
        .DataWidth( DataWidth+2 )
    ) i_fpcomp (
        .operand_a_i( flopoco_fp_a ),
        .operand_b_i( flopoco_fp_b ),

        .a_lt_b_o( a_lt_b_o ),
        .a_eq_b_o( a_eq_b_o ),
        .a_gt_b_o( a_gt_b_o ),
        .a_le_b_o( a_le_b_o ),
        .a_ge_b_o( a_ge_b_o ),

        .unordered_o( unordered_o )
    );

endmodule : IEEEComp
