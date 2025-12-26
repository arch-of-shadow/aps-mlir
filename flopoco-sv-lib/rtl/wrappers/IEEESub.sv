// Copyright 2025
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco IEEE Floating Point Subtractor unit
// The result will be valid after Latency cycles
module IEEESub #(
    /// Width of the operands (16, 32, or 64)
    parameter int unsigned DataWidth = 32,

    /// Latency of the unit
    parameter int unsigned Latency = 1,

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

    // Half precision Sub
    if (DataWidth == 16) begin : gen_sub_h
        if (Latency == 1) begin : gen_latency_h_1
            IEEESub_H1 i_sub_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            IEEESub_H2 i_sub_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            IEEESub_H3 i_sub_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            IEEESub_H4 i_sub_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_h_5
            IEEESub_H5 i_sub_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("IEEESub: Unsupported Latency %0d for DataWidth %0d", Latency,
                DataWidth);
        end : gen_unsupported_h_latency
    end : gen_sub_h
    // Single precision Sub
    else if (DataWidth == 32) begin : gen_sub_s
        if (Latency == 1) begin : gen_latency_s_1
            IEEESub_S1 i_sub_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_s_2
            IEEESub_S2 i_sub_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_s_3
            IEEESub_S3 i_sub_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            IEEESub_S4 i_sub_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            IEEESub_S5 i_sub_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_s_6
            IEEESub_S6 i_sub_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("IEEESub: Unsupported Latency %0d for DataWidth %0d", Latency,
                DataWidth);
        end : gen_unsupported_s_latency
    end : gen_sub_s
    else begin : gen_unsupported
        initial $error("IEEESub: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : IEEESub
