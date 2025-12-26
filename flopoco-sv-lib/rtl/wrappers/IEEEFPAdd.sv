// Copyright 2025
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco IEEE Floating Point Adder unit
// The result will be valid after Latency cycles
module IEEEFPAdd #(
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

    // Half precision Add
    if (DataWidth == 16) begin : gen_add_h
        if (Latency == 1) begin : gen_latency_h_1
            IEEEFPAdd_H1 i_add_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            IEEEFPAdd_H2 i_add_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            IEEEFPAdd_H3 i_add_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            IEEEFPAdd_H4 i_add_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_h_5
            IEEEFPAdd_H5 i_add_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("IEEEFPAdd: Unsupported Latency %0d for DataWidth %0d", Latency,
                DataWidth);
        end : gen_unsupported_h_latency
    end : gen_add_h
    // Single precision Add
    else if (DataWidth == 32) begin : gen_add_s
        if (Latency == 1) begin : gen_latency_s_1
            IEEEFPAdd_S1 i_add_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_s_2
            IEEEFPAdd_S2 i_add_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_s_3
            IEEEFPAdd_S3 i_add_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            IEEEFPAdd_S4 i_add_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            IEEEFPAdd_S5 i_add_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_s_6
            IEEEFPAdd_S6 i_add_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("IEEEFPAdd: Unsupported Latency %0d for DataWidth %0d", Latency,
                DataWidth);
        end : gen_unsupported_s_latency
    end : gen_add_s
    else begin : gen_unsupported
        initial $error("IEEEFPAdd: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : IEEEFPAdd
