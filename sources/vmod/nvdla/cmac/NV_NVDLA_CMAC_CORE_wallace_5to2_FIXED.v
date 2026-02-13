// ================================================================
// NVDLA Open Source Project
// 
// Copyright(c) 2016 - 2017 NVIDIA Corporation.  Licensed under the
// NVDLA Open Hardware License; Check "LICENSE" which comes with 
// this distribution for more information.
// ================================================================

// File Name: NV_NVDLA_CMAC_CORE_wallace_5to2_FIXED.v
//
// Wallace Tree 5:2 Reducer - MATCHES NV_DW02_tree INTERFACE
// Takes concatenated input like DW02_tree, produces sum + carry
//
// Based on NV_DW02_tree algorithm but uses structural Wallace tree
//
`timescale 1ns/1ps

module NV_NVDLA_CMAC_CORE_wallace_5to2_FIXED (
  INPUT,
  OUT0,
  OUT1
);

parameter num_inputs = 5;
parameter input_width = 24;

input  [num_inputs*input_width-1:0] INPUT;  // Concatenated inputs (like DW02_tree)
output [input_width-1:0] OUT0;               // Sum
output [input_width-1:0] OUT1;               // Carry

//==========================================================
// Unpack concatenated inputs into array
//==========================================================
wire [input_width-1:0] pp [0:num_inputs-1];

genvar i;
generate
  for (i = 0; i < num_inputs; i = i + 1) begin : unpack
    assign pp[i] = INPUT[i*input_width +: input_width];
  end
endgenerate

//==========================================================
// Wallace Tree 5:2 Reduction  
// Uses 3-level CSA32-only tree (matching NV_DW02_tree topology)
//==========================================================

// Level 0: CSA32 reduces pp[0:2] → sum0, carry0
wire [input_width-1:0] level0_sum;
wire [input_width-1:0] level0_carry;

NV_NVDLA_CMAC_CORE_csa32 #(.WIDTH(input_width)) u_level0_csa32 (
  .in0   (pp[0]),
  .in1   (pp[1]),
  .in2   (pp[2]),
  .sum   (level0_sum),
  .carry (level0_carry)
);

// Now have: level0_sum, level0_carry, pp[3], pp[4] = 4 values

// Level 1: CSA32 reduces sum0, carry0, pp[3] → sum1, carry1
wire [input_width-1:0] level1_sum;
wire [input_width-1:0] level1_carry;

NV_NVDLA_CMAC_CORE_csa32 #(.WIDTH(input_width)) u_level1_csa32 (
  .in0   (level0_sum),
  .in1   (level0_carry),
  .in2   (pp[3]),
  .sum   (level1_sum),
  .carry (level1_carry)
);

// Now have: level1_sum, level1_carry, pp[4] = 3 values

// Level 2: CSA32 reduces sum1, carry1, pp[4] → final OUT0, OUT1
NV_NVDLA_CMAC_CORE_csa32 #(.WIDTH(input_width)) u_level2_csa32 (
  .in0   (level1_sum),
  .in1   (level1_carry),
  .in2   (pp[4]),
  .sum   (OUT0),
  .carry (OUT1)
);

endmodule
