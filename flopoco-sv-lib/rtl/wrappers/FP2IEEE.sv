// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco Floating Point to IEEE 754 converter
module FP2IEEE #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Dependent parameter, do **not** overwrite.
    parameter type ieee_t = logic [DataWidth-1:0],
    parameter type fp_t   = logic [DataWidth+1:0]
) (
    /// Flopoco Floating Point
    input fp_t fp_i,

    /// IEEE 754 Floating Point
    output ieee_t ieee_o
);

    // Half precision Exp
    if (DataWidth == 16) begin : gen_conv_h
        FP2IEEE_H0 i_conv_h (
            .clk( 1'b0 ),
            .rst( 1'b0 ),

            .X( fp_i   ),
            .R( ieee_o )
        );
    end : gen_conv_h
    // Single precision Exp
    else if (DataWidth == 32) begin : gen_conv_s
        FP2IEEE_S0 i_conv_s (
            .clk( 1'b0 ),
            .rst( 1'b0 ),

            .X( fp_i   ),
            .R( ieee_o )
        );
    end : gen_conv_s
    else begin : gen_unsupported
        initial $error("FP2IEEE: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : FP2IEEE
