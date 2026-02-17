// ================================================================
// NVDLA Open Source Project
// 
// Copyright(c) 2016 - 2017 NVIDIA Corporation.  Licensed under the
// NVDLA Open Hardware License; Check "LICENSE" which comes with 
// this distribution for more information.
// ================================================================

// File Name: NV_NVDLA_CMAC_CORE_csa42.v

module NV_NVDLA_CMAC_CORE_csa42 #(
  parameter WIDTH = 24
)(
  input  [WIDTH-1:0] in0,
  input  [WIDTH-1:0] in1,
  input  [WIDTH-1:0] in2,
  input  [WIDTH-1:0] in3,
  input  [WIDTH-1:0] cin,
  output [WIDTH-1:0] sum,
  output [WIDTH-1:0] carry,
  output [WIDTH-1:0] cout
);

// Fill your implementation here

assign sum = {WIDTH{1'b0}};
assign carry = {WIDTH{1'b0}};
assign cout = {WIDTH{1'b0}};

endmodule
