// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco IEEE Floating Point Multiply-Add (FMA) unit
// Computes (A * B) + C
// The result will be valid after Latency cycles
module IEEEFMA #(
    /// Width of the operands
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
    input data_t operand_c_i,
    input logic  negate_a_i,
    input logic  negate_c_i,

    /// Output result
    output data_t result_o
);

    // Half precision FMA
    if (DataWidth == 16) begin : gen_fma_h
        if (Latency == 1) begin : gen_latency_h_1
            IEEEFMA_H1 i_fma_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_h_2
            IEEEFMA_H2 i_fma_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_h_3
            IEEEFMA_H3 i_fma_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 5) begin : gen_latency_h_5
            IEEEFMA_H5 i_fma_h (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else begin : gen_unsupported_h_latency
            initial $error("IEEEFMA: Unsupported Latency %0d for DataWidth %0d", Latency,
                DataWidth);
        end : gen_unsupported_h_latency
    end : gen_fma_h
    // Single precision FMA
    if (DataWidth == 32) begin : gen_fma_s
        if (Latency == 1) begin : gen_latency_s_1
            IEEEFMA_S1 i_fma_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 2) begin : gen_latency_s_2
            IEEEFMA_S2 i_fma_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 3) begin : gen_latency_s_3
            IEEEFMA_S3 i_fma_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 4) begin : gen_latency_s_4
            IEEEFMA_S4 i_fma_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else if (Latency == 6) begin : gen_latency_s_6
            IEEEFMA_S6 i_fma_s (
                .clk( clk_i   ),
                .rst( !rst_ni ),

                .A( operand_a_i ),
                .B( operand_b_i ),
                .C( operand_c_i ),

                .negateAB( negate_a_i ),
                .negateC ( negate_c_i ),

                .RndMode( 'd0 ), // Non Functional

                .R( result_o )
            );
        end else begin : gen_unsupported_s_latency
            initial $error("IEEEFMA: Unsupported Latency %0d for DataWidth %0d", Latency,
                DataWidth);
        end : gen_unsupported_s_latency
    end : gen_fma_s
    else begin : gen_unsupported
        initial $error("IEEEFMA: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : IEEEFMA
