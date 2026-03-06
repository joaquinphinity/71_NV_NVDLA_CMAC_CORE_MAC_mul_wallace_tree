# Synthesis script for NV_NVDLA_CMAC_CORE_MAC_mul (task 71 - Wallace Tree)
# DESIGNWARE_NOEXIST selects the NVDLA-provided NV_DW02_tree simulation model
# instead of the DesignWare DW02_tree IP (which Yosys cannot resolve).
# Note: if the agent's solution fully replaces DW02_tree with custom modules,
# DESIGNWARE_NOEXIST is still needed for the ifdef guards that remain in mul.v.
verilog_defaults -define DESIGNWARE_NOEXIST
read_verilog sources/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_MAC_mul.v
read_verilog sources/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_csa32.v
read_verilog sources/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_csa42.v
read_verilog sources/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_wallace_4to2.v
read_verilog sources/vmod/nvdla/cmac/NV_NVDLA_CMAC_CORE_wallace_5to2.v
read_verilog sources/vmod/vlibs/NV_DW02_tree.v
hierarchy -check -top NV_NVDLA_CMAC_CORE_MAC_mul
proc; opt; memory; opt
techmap; opt
abc; opt
stat
