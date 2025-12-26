// Copyright 2025
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the FloPoCo FixSinCos unit
/// Computes sin(pi*X) and cos(pi*X) for X in [-1, 1)
/// Input/Output are fixed-point Q1.N format where N = Precision
/// The result will be valid after Latency cycles
module FixSinCos #(
    /// Precision (number of fractional bits)
    /// 24 for single-precision equivalent, 11 for half-precision
    parameter int unsigned Precision = 24,

    /// Latency of the unit (pipeline stages)
    parameter int unsigned Latency = 2,

    /// Width = Precision + 1 (sign bit)
    parameter int unsigned Width = Precision + 1
) (
    /// Clock and Reset
    input  logic clk_i,
    input  logic rst_ni,

    /// Input: fixed-point value in [-1, 1), Q1.Precision format
    input  logic [Width-1:0] x_i,

    /// Output: sin(pi * x_i)
    output logic [Width-1:0] sin_o,

    /// Output: cos(pi * x_i)
    output logic [Width-1:0] cos_o
);

    // 24-bit precision (single-precision equivalent)
    if (Precision == 24) begin : gen_sincos_24
        if (Latency == 1) begin : gen_latency_1
            FixSinCos_24_S1 i_sincos (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X  ( x_i     ),
                .S  ( sin_o   ),
                .C  ( cos_o   )
            );
        end else if (Latency == 2) begin : gen_latency_2
            FixSinCos_24_S2 i_sincos (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X  ( x_i     ),
                .S  ( sin_o   ),
                .C  ( cos_o   )
            );
        end else if (Latency == 3) begin : gen_latency_3
            FixSinCos_24_S3 i_sincos (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X  ( x_i     ),
                .S  ( sin_o   ),
                .C  ( cos_o   )
            );
        end else if (Latency == 4) begin : gen_latency_4
            FixSinCos_24_S4 i_sincos (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X  ( x_i     ),
                .S  ( sin_o   ),
                .C  ( cos_o   )
            );
        end else begin : gen_unsupported_latency
            initial $error("FixSinCos: Unsupported Latency %0d for Precision %0d", Latency, Precision);
        end
    end : gen_sincos_24

    // 11-bit precision (half-precision equivalent)
    else if (Precision == 11) begin : gen_sincos_11
        if (Latency == 1) begin : gen_latency_1
            FixSinCos_11_S1 i_sincos (
                .clk( clk_i   ),
                .rst( !rst_ni ),
                .X  ( x_i     ),
                .S  ( sin_o   ),
                .C  ( cos_o   )
            );
        end else begin : gen_unsupported_latency
            initial $error("FixSinCos: Unsupported Latency %0d for Precision %0d", Latency, Precision);
        end
    end : gen_sincos_11

    else begin : gen_unsupported
        initial $error("FixSinCos: Unsupported Precision %0d", Precision);
    end : gen_unsupported

endmodule : FixSinCos
