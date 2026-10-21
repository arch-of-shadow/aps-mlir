# Pre-layout IHP SG13G2 TT 1.20 V / 25 C. Liberty units: ns, pF.
# Target 4 ns, not a relaxed clock inferred from the measured critical path.
read_liberty $env(LIBERTY)
read_verilog $env(NETLIST)
link_design $env(TOP)
set cp clock
set rp reset
if {[info exists env(CLOCK_PORT)]} {set cp $env(CLOCK_PORT)}
if {[info exists env(RESET_PORT)]} {set rp $env(RESET_PORT)}
create_clock -name core -period 4.0 [get_ports $cp]
set_propagated_clock [all_clocks]
set_clock_uncertainty -setup 0.10 [all_clocks]
set_clock_uncertainty -hold 0.05 [all_clocks]
set inputs {}
foreach p [all_inputs] {
  if {$p ne [lindex [get_ports $cp] 0]} { lappend inputs $p }
}
set_input_delay -max 0.20 -clock core $inputs
set_input_delay -min 0.10 -clock core $inputs
set_input_transition 0.10 $inputs
set_output_delay -max 0.20 -clock core [all_outputs]
set_output_delay -min 0.00 -clock core [all_outputs]
set_load 0.02 [all_outputs]
set_case_analysis 0 [get_ports $rp]
check_setup -verbose
puts "ACCEPTANCE_SETUP"
report_checks -path_delay max -group_count 5 -digits 4
report_worst_slack -digits 4
puts "ACCEPTANCE_HOLD"
report_checks -path_delay min -group_count 5 -digits 4
report_tns
exit
