// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Signed integer to IEEE 754 Floating Point converter
// The result will be valid after Latency cycles
module INT2IEEE #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Latency of the unit
    parameter int unsigned Latency = 0,

    /// Dependent parameter, do **not** overwrite.
    parameter type data_t = logic [DataWidth-1:0]
) (
    /// Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    /// Input operands
    input data_t int_i,

    /// Output result
    output data_t ieee_o
);
    // Flopoco Floating Point
    logic [DataWidth+1:0] flopoco_fp;

    // Convert INT to Flopoco FP
    INT2FP #(
        .DataWidth( DataWidth ),
        .Latency  ( Latency   )
    ) i_int2fp (
        .clk_i ( clk_i  ),
        .rst_ni( rst_ni ),

        .int_i( int_i ),

        .fp_o( flopoco_fp )
    );

    // Convert Flopoco FP to IEEE 754
    FP2IEEE #(
        .DataWidth( DataWidth )
    ) i_fp2ieee (
        .fp_i  ( flopoco_fp ),
        .ieee_o( ieee_o     )
    );

endmodule : INT2IEEE
