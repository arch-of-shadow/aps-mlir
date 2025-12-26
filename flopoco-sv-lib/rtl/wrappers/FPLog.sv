// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco Floating Point Logarithm
// Computes log(x)
// The result will be valid after Latency cycles
module FPLog #(
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

    // Half precision Logarithm
    if (DataWidth == 18) begin : gen_log_h
        if (Latency == 1) begin : gen_latency_h_1
            FPLog_H1 i_log_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            FPLog_H2 i_log_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            FPLog_H3 i_log_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            FPLog_H4 i_log_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_h_5
            FPLog_H5 i_log_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 7) begin : gen_latency_h_7
            FPLog_H7 i_log_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("FPLog: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_h_latency
    end : gen_log_h
    // Single precision Logarithm
    if (DataWidth == 34) begin : gen_log_s
        if (Latency == 2) begin : gen_latency_s_2
            FPLog_S2 i_log_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_s_3
            FPLog_S3 i_log_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            FPLog_S4 i_log_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            FPLog_S5 i_log_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_s_6
            FPLog_S6 i_log_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else if (Latency == 9) begin : gen_latency_s_9
            FPLog_S9 i_log_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .X( operand_x_i ),

                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("FPLog: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_s_latency
    end : gen_log_s
    else begin : gen_unsupported
        initial $error("FPLog: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : FPLog
