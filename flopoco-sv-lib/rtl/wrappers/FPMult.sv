// Copyright 2025
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco Floating Point Multiply unit (FloPoCo internal format)
// Computes A * B
// The result will be valid after Latency cycles
module FPMult #(
    /// Width of the operands (FloPoCo internal format: DataWidth = IEEE_width + 2)
    parameter int unsigned DataWidth = 34,

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

    // Half precision Multiply (FloPoCo internal format: 18 bits)
    if (DataWidth == 18) begin : gen_mul_h
        if (Latency == 1) begin : gen_latency_h_1
            FPMult_H1 i_mul_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            FPMult_H2 i_mul_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            FPMult_H3 i_mul_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_h_4
            FPMult_H4 i_mul_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_h_5
            FPMult_H5 i_mul_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("FPMult: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_h_latency
    end : gen_mul_h
    // Single precision Multiply (FloPoCo internal format: 34 bits)
    else if (DataWidth == 34) begin : gen_mul_s
        if (Latency == 1) begin : gen_latency_s_1
            FPMult_S1 i_mul_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_s_2
            FPMult_S2 i_mul_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_s_3
            FPMult_S3 i_mul_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            FPMult_S4 i_mul_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_s_5
            FPMult_S5 i_mul_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_s_6
            FPMult_S6 i_mul_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X( operand_a_i ),
                .Y( operand_b_i ),
                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("FPMult: Unsupported Latency %0d for DataWidth %0d", Latency, DataWidth);
        end : gen_unsupported_s_latency
    end : gen_mul_s
    else begin : gen_unsupported
        initial $error("FPMult: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : FPMult
