// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// IEEE 754 Floating Point to signed integer converter
// The result will be valid after Latency cycles
module IEEE2INT #(
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
    input data_t ieee_i,

    /// Output result
    output data_t int_o,
    output logic  overflow_o
);
    // Flopoco Floating Point
    logic [DataWidth+1:0] flopoco_fp;

    // Convert IEEE 754 to Flopoco FP
    IEEE2FP #(
        .DataWidth( DataWidth )
    ) i_ieee2fp (
        .ieee_i( ieee_i     ),
        .fp_o  ( flopoco_fp )
    );

    // Convert Flopoco FP to INT
    FP2INT #(
        .DataWidth( DataWidth ),
        .Latency  ( Latency   )
    ) i_fp2int (
        .clk_i ( clk_i  ),
        .rst_ni( rst_ni ),

        .fp_i ( flopoco_fp ),
        .int_o( int_o      ),

        .overflow_o( overflow_o )
    );

endmodule : IEEE2INT
