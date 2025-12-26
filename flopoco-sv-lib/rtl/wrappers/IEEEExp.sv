// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco IEEE Exponential (Exp) unit
// Computes exp(A)
// The result will be valid after Latency cycles
module IEEEExp #(
    /// Width of the operands
    parameter int unsigned DataWidth = 32,

    /// Latency of the unit
    parameter int unsigned Latency = 3,

    /// Dependent parameter, do **not** overwrite.
    parameter type data_t = logic [DataWidth-1:0]
) (
    /// Clock and Reset
    input logic clk_i,
    input logic rst_ni,

    /// Input operands
    input data_t operand_a_i,

    /// Output result
    output data_t result_o
);

    // Half precision Exp
    if (DataWidth == 16) begin : gen_exp_h
        if (Latency == 1) begin : gen_latency_h_1
            IEEEExp_H1 i_exp_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            IEEEExp_H2 i_exp_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            IEEEExp_H3 i_exp_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            IEEEExp_H4 i_exp_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_h_6
            IEEEExp_H6 i_exp_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("EXP: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_h_latency
    end : gen_exp_h
    // Single precision Exp
    if (DataWidth == 32) begin : gen_exp_s
        if (Latency == 3) begin : gen_latency_s_3
            IEEEExp_S3 i_exp_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            IEEEExp_S4 i_exp_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            IEEEExp_S5 i_exp_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_s_6
            IEEEExp_S6 i_exp_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 7) begin : gen_latency_s_7
            IEEEExp_S7 i_exp_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else if (Latency == 9) begin : gen_latency_s_9
            IEEEExp_S9 i_exp_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_a_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("EXP: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_s_latency
    end : gen_exp_s
    else begin : gen_unsupported
        initial $error("IEEEExp: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : IEEEExp
