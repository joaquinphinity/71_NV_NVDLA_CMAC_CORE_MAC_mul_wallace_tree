// ================================================================
// NVDLA Open Source Project
// 
// Copyright(c) 2016 - 2017 NVIDIA Corporation.  Licensed under the
// NVDLA Open Hardware License; Check "LICENSE" which comes with 
// this distribution for more information.
// ================================================================

// File Name: NV_NVDLA_CMAC_CORE_csa32.v
//
// 3:2 Carry-Save Adder (Full Adder)
// Reduces 3 inputs to 2 outputs (sum + carry) without carry propagation
//
`timescale 1ns/1ps

module NV_NVDLA_CMAC_CORE_csa32 #(
  parameter WIDTH = 24
)(
  input  [WIDTH-1:0] in0,
  input  [WIDTH-1:0] in1,
  input  [WIDTH-1:0] in2,
  output [WIDTH-1:0] sum,
  output [WIDTH-1:0] carry
);

//==========================================================
// 3:2 Carry-Save Adder Logic
//==========================================================
// Sum: XOR of all three inputs
// Carry: Majority function, left-shifted by 1

assign sum = in0 ^ in1 ^ in2;

// Carry = (in0&in1) | (in1&in2) | (in0&in2), shifted left by 1
assign carry = {((in0[WIDTH-2:0] & in1[WIDTH-2:0]) | 
                 (in1[WIDTH-2:0] & in2[WIDTH-2:0]) | 
                 (in0[WIDTH-2:0] & in2[WIDTH-2:0])), 1'b0};

endmodule
