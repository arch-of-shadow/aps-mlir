// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco Floating Point Divide unit
// Computes A / B
// The result will be valid after Latency cycles
module FPDiv #(
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
    input data_t operand_a_i,
    input data_t operand_b_i,

    /// Output result
    output data_t result_o
);

    // Half precision Divide
    if (DataWidth == 18) begin : gen_div_h
        if (Latency == 1) begin : gen_latency_h_1
            FPDiv_H1 i_div_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            FPDiv_H2 i_div_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            FPDiv_H3 i_div_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            FPDiv_H4 i_div_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_h_6
            FPDiv_H6 i_div_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("FPDiv: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_h_latency
    end : gen_div_h
    // Single precision Divide
    if (DataWidth == 34) begin : gen_div_s
        if (Latency == 2) begin : gen_latency_s_2
            FPDiv_S2 i_div_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            FPDiv_S4 i_div_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            FPDiv_S5 i_div_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 7) begin : gen_latency_s_7
            FPDiv_S7 i_div_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 8) begin : gen_latency_s_8
            FPDiv_S8 i_div_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else if (Latency == 12) begin : gen_latency_s_12
            FPDiv_S12 i_div_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),
                .Y( operand_b_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("FPDiv: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_s_latency
    end : gen_div_s
    else begin : gen_unsupported
        initial $error("FPDiv: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : FPDiv
