// Copyright 2025 Tobias Senti
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

/// Wrapper for the Flopoco Floating Point comparator
module FPComp #(
    /// Width of the operands
    parameter int unsigned DataWidth = 34,

    /// Dependent parameter, do **not** overwrite.
    parameter type fp_t   = logic [DataWidth-1:0]
) (
    /// Input operands
    input fp_t operand_a_i,
    input fp_t operand_b_i,

    /// Output result
    output logic a_lt_b_o,
    output logic a_eq_b_o,
    output logic a_gt_b_o,
    output logic a_le_b_o,
    output logic a_ge_b_o,
    output logic unordered_o
);

    // Half precision Comparator
    if (DataWidth == 18) begin : gen_comp_h
        FPComp_H0 i_comp_h (
            .clk( 1'b0 ),
            .rst( 1'b0 ),

            .X( operand_a_i ),
            .Y( operand_b_i ),

            .XltY( a_lt_b_o ),
            .XeqY( a_eq_b_o ),
            .XgtY( a_gt_b_o ),
            .XleY( a_le_b_o ),
            .XgeY( a_ge_b_o ),

            .unordered( unordered_o )
        );
    end : gen_comp_h
    // Single precision Comparator
    if (DataWidth == 34) begin : gen_comp_s
        FPComp_S0 i_comp_s (
            .clk( 1'b0 ),
            .rst( 1'b0 ),

            .X( operand_a_i ),
            .Y( operand_b_i ),

            .XltY( a_lt_b_o ),
            .XeqY( a_eq_b_o ),
            .XgtY( a_gt_b_o ),
            .XleY( a_le_b_o ),
            .XgeY( a_ge_b_o ),

            .unordered( unordered_o )
        );
    end : gen_comp_s
    else begin : gen_unsupported
        initial $error("FPComp: Unsupported DataWidth %0d", DataWidth);
    end : gen_unsupported

endmodule : FPComp
