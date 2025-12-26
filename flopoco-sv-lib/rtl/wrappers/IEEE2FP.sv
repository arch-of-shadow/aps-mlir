// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco IEEE 754 to Flopoco Floating Point converter
module IEEE2FP #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Dependent parameter, do **not** overwrite.
    parameter type ieee_t = logic [DataWidth-1:0],
    parameter type fp_t   = logic [DataWidth+1:0]
) (
    /// IEEE 754 Floating Point
    input ieee_t ieee_i,

    /// Flopoco Floating Point
    output fp_t fp_o
);

    // Half precision
    if (DataWidth == 16) begin : gen_conv_h
        IEEE2FP_H0 i_conv_h (
            .clk( 1'b0 ),
            .rst( 1'b0 ),

            .X( ieee_i ),
            .R( fp_o   )
        );
    end : gen_conv_h
    // Single precision
    else if (DataWidth == 32) begin : gen_conv_s
        IEEE2FP_S0 i_conv_s (
            .clk( 1'b0 ),
            .rst( 1'b0 ),

            .X( ieee_i ),
            .R( fp_o   )
        );
    end : gen_conv_s
    else begin : gen_unsupported
        initial $error("IEEE2FP: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : IEEE2FP
