// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco Floating Point Square Root
// Computes sqrt(x)
// The result will be valid after Latency cycles
module FPSqrt #(
    /// Width of the operands
    parameter int unsigned DataWidth = 34,

    /// Latency of the unit
    parameter int unsigned Latency = 2,

    /// Dependent parameter, do **not** overwrite.
    parameter type data_t = logic [DataWidth-1:0]
) (
    /// Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    /// Input operands
    input data_t operand_x_i,

    /// Output result
    output data_t result_o
);

    // Half precision Square Root
    if (DataWidth == 18) begin : gen_sqrt_h
        if (Latency == 1) begin : gen_latency_h_1
            FPSqrt_H1 i_sqrt_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            FPSqrt_H2 i_sqrt_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            FPSqrt_H3 i_sqrt_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            FPSqrt_H4 i_sqrt_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_h_5
            FPSqrt_H5 i_sqrt_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("FPSqrt: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_h_latency
    end : gen_sqrt_h
    // Single precision Square Root
    if (DataWidth == 34) begin : gen_sqrt_s
        if (Latency == 2) begin : gen_latency_s_2
            FPSqrt_S2 i_sqrt_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            FPSqrt_S4 i_sqrt_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            FPSqrt_S5 i_sqrt_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 7) begin : gen_latency_s_7
            FPSqrt_S7 i_sqrt_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 9) begin : gen_latency_s_9
            FPSqrt_S9 i_sqrt_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 12) begin : gen_latency_s_12
            FPSqrt_S12 i_sqrt_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("FPSqrt: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_s_latency
    end : gen_sqrt_s
    else begin : gen_unsupported
        initial $error("FPSqrt: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : FPSqrt
