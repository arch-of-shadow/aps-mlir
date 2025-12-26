yosys tee -q -o "modules.rpt" ls
set fp [open "modules.rpt" r]
set file_data [read $fp]
close $fp
set modules [split $file_data "\n"]
set i 0
foreach module $modules {
    if {$i < 3} {
        incr i
        continue
    }
    set module [string trim $module]
    if {$module == ""} {
        continue
    }
    exec sed -i -e "s/${module}/${module}${suffix}/g" $file_name
}
