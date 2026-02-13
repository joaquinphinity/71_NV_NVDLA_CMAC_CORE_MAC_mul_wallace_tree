// ================================================================
// NVDLA Open Source Project
// 
// Copyright(c) 2016 - 2017 NVIDIA Corporation.  Licensed under the
// NVDLA Open Hardware License; Check "LICENSE" which comes with 
// this distribution for more information.
// ================================================================

// File Name: NV_DW02_tree.v
`timescale 1ns/1ps
module NV_DW02_tree( INPUT, OUT0, OUT1 );
parameter num_inputs = 8;
parameter input_width = 8;
input [num_inputs*input_width-1 : 0]	INPUT;
output [input_width-1:0]		OUT0, OUT1;

//impliment functionality
endmodule

