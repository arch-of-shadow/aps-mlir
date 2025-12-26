// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// IEEE 754 Floating Point division
module IEEEDiv #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Latency of the unit
    parameter int unsigned Latency = 2,

    /// Dependent parameter, do **not** overwrite.
    parameter type ieee_t = logic [DataWidth-1:0]
) (
    /// Clock and reset
    input logic clk_i,
    input logic rst_ni,

    /// Input operands
    input ieee_t operand_a_i,
    input ieee_t operand_b_i,

    /// Output result
    output ieee_t result_o
);
    // Flopoco Floating Point
    logic [DataWidth+1:0] flopoco_fp_a, flopoco_fp_b, flopoco_fp_result;

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

    // Divider
    FPDiv #(
        .DataWidth( DataWidth+2 ),
        .Latency  ( Latency     )
    ) i_fpdiv (
        .clk_i ( clk_i  ),
        .rst_ni( rst_ni ),

        .operand_a_i( flopoco_fp_a ),
        .operand_b_i( flopoco_fp_b ),

        .result_o( flopoco_fp_result )
    );

    // Convert Flopoco FP to IEEE 754
    FP2IEEE #(
        .DataWidth( DataWidth )
    ) i_fp2ieee (
        .fp_i  ( flopoco_fp_result ),
        .ieee_o( result_o          )
    );

endmodule : IEEEDiv
