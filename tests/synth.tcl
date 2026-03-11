# Synthesis script for NV_NVDLA_CMAC_CORE_MAC_mul (task 71 - Wallace Tree)
# DESIGNWARE_NOEXIST is needed for the ifdef guards in mul.v even when the agent
# has fully replaced NV_DW02_tree with custom structural modules.
# NV_DW02_tree.v is intentionally NOT read: agents must replace all NV_DW02_tree
# instances with synthesizable structural RTL. Any lingering NV_DW02_tree
# instantiation will cause 'hierarchy -check' to fail (module not found).
verilog_defaults -define DESIGNWARE_NOEXIST
# Read all agent-created files from cmac/ — glob avoids hardcoded filenames
# so any .v file the agent places in cmac/ is included automatically.
foreach f [glob -nocomplain sources/vmod/nvdla/cmac/*.v] {
    read_verilog $f
}
hierarchy -check -top NV_NVDLA_CMAC_CORE_MAC_mul
proc; opt; memory; opt
techmap; opt
abc; opt
stat
