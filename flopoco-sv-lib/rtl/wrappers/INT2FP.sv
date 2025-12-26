// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco signed integer to Floating Point converter
// The result will be valid after Latency cycles
module INT2FP #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Latency of the unit
    parameter int unsigned Latency = 0,

    /// Dependent parameter, do **not** overwrite.
    parameter type fp_t   = logic [DataWidth+1:0],
    parameter type data_t = logic [DataWidth-1:0]
) (
    /// Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    /// Input operands
    input data_t int_i,

    /// Output result
    output fp_t fp_o
);

    // Half precision
    if (DataWidth == 16) begin : gen_conv_h
        if (Latency == 0) begin : gen_latency_h_0
            Fix2FP_H0 i_conv_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .I( int_i ),

                .O ( fp_o )
            );
        end else if (Latency == 1) begin : gen_latency_h_1
            Fix2FP_H1 i_conv_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .I( int_i ),

                .O ( fp_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            Fix2FP_H2 i_conv_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .I( int_i ),

                .O ( fp_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("INT2FP: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_h_latency
    end : gen_conv_h
    // Single precision
    if (DataWidth == 32) begin : gen_conv_s
        if (Latency == 0) begin : gen_latency_s_0
            Fix2FP_S0 i_conv_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .I( int_i ),

                .O ( fp_o )
            );
        end else if (Latency == 1) begin : gen_latency_s_1
            Fix2FP_S1 i_conv_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .I( int_i ),

                .O ( fp_o )
            );
        end else if (Latency == 2) begin : gen_latency_s_2
            Fix2FP_S2 i_conv_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .I( int_i ),

                .O ( fp_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("INT2FP: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_s_latency
    end : gen_conv_s
    else begin : gen_unsupported
        initial $error("INT2FP: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : INT2FP
