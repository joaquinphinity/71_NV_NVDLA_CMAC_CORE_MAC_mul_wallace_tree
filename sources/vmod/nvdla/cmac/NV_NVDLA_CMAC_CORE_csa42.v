// ================================================================
// NVDLA Open Source Project
// 
// Copyright(c) 2016 - 2017 NVIDIA Corporation.  Licensed under the
// NVDLA Open Hardware License; Check "LICENSE" which comes with 
// this distribution for more information.
// ================================================================

// File Name: NV_NVDLA_CMAC_CORE_csa42.v
//
// 4:2 Compressor (5:3 with carry chains)
// Reduces 4 inputs + 1 carry-in to 2 outputs + 1 carry-out
// Uses two 3:2 CSAs (full adders) in sequence
//
`timescale 1ns/1ps

module NV_NVDLA_CMAC_CORE_csa42 #(
  parameter WIDTH = 24
)(
  input  [WIDTH-1:0] in0,
  input  [WIDTH-1:0] in1,
  input  [WIDTH-1:0] in2,
  input  [WIDTH-1:0] in3,
  input  [WIDTH-1:0] cin,      // Carry from previous compressor
  output [WIDTH-1:0] sum,
  output [WIDTH-1:0] carry,
  output [WIDTH-1:0] cout      // Carry to next compressor (independent)
);

//==========================================================
// 4:2 Compressor Logic using two Full Adders
//==========================================================

// First FA: Compress in0, in1, in2 → sum1, carry1
wire [WIDTH-1:0] fa1_sum;
wire [WIDTH-1:0] fa1_carry;

assign fa1_sum   = in0 ^ in1 ^ in2;
assign fa1_carry = (in0 & in1) | (in1 & in2) | (in0 & in2);

// Second FA: Compress fa1_sum, in3, cin → final sum, carry
wire [WIDTH-1:0] fa2_sum;
wire [WIDTH-1:0] fa2_carry;

assign fa2_sum   = fa1_sum ^ in3 ^ cin;
assign fa2_carry = (fa1_sum & in3) | (in3 & cin) | (fa1_sum & cin);

//==========================================================
// Outputs (carries left-shifted by 1)
//==========================================================

assign sum   = fa2_sum;
assign carry = {fa2_carry[WIDTH-2:0], 1'b0};  // Primary carry (from FA2)
assign cout  = {fa1_carry[WIDTH-2:0], 1'b0};  // Independent carry (from FA1)

endmodule
