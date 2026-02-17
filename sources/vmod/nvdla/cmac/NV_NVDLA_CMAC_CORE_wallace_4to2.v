// ================================================================
// NVDLA Open Source Project
// 
// Copyright(c) 2016 - 2017 NVIDIA Corporation.  Licensed under the
// NVDLA Open Hardware License; Check "LICENSE" which comes with 
// this distribution for more information.
// ================================================================

// File Name: NV_NVDLA_CMAC_CORE_wallace_4to2.v
//
// Wallace Tree 4:2 Reducer - MATCHES NV_DW02_tree INTERFACE
// Takes concatenated input like DW02_tree, produces sum + carry
//
`timescale 1ns/1ps

module NV_NVDLA_CMAC_CORE_wallace_4to2 (
  INPUT,
  OUT0,
  OUT1
);

parameter num_inputs = 4;
parameter input_width = 32;

input  [num_inputs*input_width-1:0] INPUT;  // Concatenated inputs
output [input_width-1:0] OUT0;               // Sum
output [input_width-1:0] OUT1;               // Carry

//==========================================================
// Unpack concatenated inputs
//==========================================================
wire [input_width-1:0] in [0:num_inputs-1];

genvar i;
generate
  for (i = 0; i < num_inputs; i = i + 1) begin : unpack
    assign in[i] = INPUT[i*input_width +: input_width];
  end
endgenerate

//==========================================================
// 4:2 Reduction using 2-level CSA32 tree (matches NV_DW02_tree)
//==========================================================

// Level 0: CSA32 reduces in[0:2] → sum0, carry0
wire [input_width-1:0] level0_sum;
wire [input_width-1:0] level0_carry;

NV_NVDLA_CMAC_CORE_csa32 #(.WIDTH(input_width)) u_level0_csa32 (
  .in0   (in[0]),
  .in1   (in[1]),
  .in2   (in[2]),
  .sum   (level0_sum),
  .carry (level0_carry)
);

// Now have: level0_sum, level0_carry, in[3] = 3 values

// Level 1: CSA32 reduces sum0, carry0, in[3] → final OUT0, OUT1
NV_NVDLA_CMAC_CORE_csa32 #(.WIDTH(input_width)) u_level1_csa32 (
  .in0   (level0_sum),
  .in1   (level0_carry),
  .in2   (in[3]),
  .sum   (OUT0),
  .carry (OUT1)
);

endmodule
