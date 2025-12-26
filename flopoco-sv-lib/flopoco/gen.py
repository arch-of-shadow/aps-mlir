#!/usr/bin/env python3
# From https://github.com/jiegec/fpu-wrappers/blob/master/fpu-wrappers/resources/flopoco/gen.py

import subprocess
import os

tasks = [
{
    'type': 'H',
    'exp': 5,
    'frac': 10
},
{
    'type': 'S',
    'exp': 8,
    'frac': 23
},
# {
#     'type': 'D',
#     'exp': 11,
#     'frac': 52
# }
]

flopoco = 'flopoco'
flopoco_411 = flopoco + '-4.1.1'

def gen_fma(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "IEEEFPFMA", f"wE={task['exp']}", f"wF={task['frac']}",
               f"name=IEEEFMA_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"IEEEFMA_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("IEEEFMA", name, task, stages)

def gen_fix2fp(frequency, task):
    # generate vhdl
    command = [flopoco_411, "Fix2FP", f"wE={task['exp']}", f"wF={task['frac']}",
               "LSB=0", f"MSB={task['exp']+task['frac']}",
               f"name=Fix2FP_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = 0
    for line in out.splitlines():
        if 'Pipeline depth' in line:
            stages = int(line.split(' ')[-1])

    # save vhdl
    name = f"Fix2FP_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("Fix2FP", name, task, stages)

def gen_fp2fix(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "FP2Fix", f"wE={task['exp']}", f"wF={task['frac']}",
               "LSB=0", f"MSB={task['exp']+task['frac']}",
               f"name=FP2Fix_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"FP2Fix_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("FP2Fix", name, task, stages)

def gen_IEEE2fp(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "InputIEEE", f"wEIn={task['exp']}", f"wFIn={task['frac']}",
               f"wEOut={task['exp']}", f"wFOut={task['frac']}",
               f"name=IEEE2FP_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"IEEE2FP_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("IEEE2FP", name, task, stages)

def gen_fp2IEEEp(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "OutputIEEE", f"wEIn={task['exp']}", f"wFIn={task['frac']}",
               f"wEOut={task['exp']}", f"wFOut={task['frac']}",
               f"name=FP2IEEE_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"FP2IEEE_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("FP2IEEE", name, task, stages)

def gen_exp(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "IEEEFPExp", f"wE={task['exp']}", f"wF={task['frac']}",
            f"name=IEEEExp_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"IEEEExp_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("IEEEExp", name, task, stages)

def gen_div(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "FPDiv", f"wE={task['exp']}", f"wF={task['frac']}",
            f"name=FPDiv_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"FPDiv_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("FPDiv", name, task, stages)

def gen_comp(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "FPComparator", f"wE={task['exp']}", f"wF={task['frac']}",
            f"name=FPComp_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"FPComp_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("FPComp", name, task, stages)

def gen_sqrt(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "FPSqrt", f"wE={task['exp']}", f"wF={task['frac']}",
            f"name=FPSqrt_{task['type']}", f"frequency={frequency}"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"FPSqrt_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("FPSqrt", name, task, stages)

def gen_log(frequency, task):
    # generate vhdl
    command = [flopoco, "allRegistersWithAsyncReset=1", "FPLog", f"wE={task['exp']}", f"wF={task['frac']}",
            f"name=FPLog_{task['type']}", f"frequency={frequency}", "method=0"]
    print(' '.join(command))
    out = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT).decode('utf-8')

    print(out)

    # parse stages from output
    stages = out.splitlines()[-1]
    print(stages)
    stages = stages.split(',')[0].split('c')[1]

    # save vhdl
    name = f"FPLog_{task['type']}{stages}"
    file = f"{name}.vhdl"
    os.rename('flopoco.vhdl', file)
    convert_to_verilog("FPLog", name, task, stages)

def convert_to_verilog(dir, name, task, stages):
    # synthesize to verilog
    os.system("rm -fp run.tcl")
    os.system(f"mkdir -p {dir}")
    script = f"""
    yosys -import;
    ghdl --std=08 -fsynopsys -fexplicit {name}.vhdl -e {dir}_{task['type']};
    hierarchy -top {dir}_{task['type']};
    set suffix _{dir}_{task['type']}{stages};
    yosys rename -top {name};
    set file_name "{dir}/{name}.v";
    write_verilog $file_name;
    source rename.tcl;
    exit;
    """
    with open('run.tcl', 'w') as f:
        f.write(script)
    os.system(f"yosys -m ghdl -C run.tcl")
    os.system("rm -f run.tcl")
    os.system("rm -f modules.rpt")

os.system("rm -rf IEEEFMA")
os.system("rm -rf Fix2FP")
os.system("rm -rf FP2Fix")
os.system("rm -rf IEEE2FP")
os.system("rm -rf FP2IEEE")
os.system("rm -rf IEEEExp")
os.system("rm -rf FPDiv")
os.system("rm -rf FPComp")
os.system("rm -rf FPSqrt")
os.system("rm -rf FPLog")

for task in tasks:
    for frequency in [100, 150, 200, 250, 300, 400]:
        gen_fma(frequency, task)
        gen_fix2fp(frequency, task)
        gen_fp2fix(frequency, task)
        gen_IEEE2fp(frequency, task)
        gen_fp2IEEEp(frequency, task)
        gen_exp(frequency, task)
        gen_div(frequency, task)
        gen_comp(frequency, task)
        gen_sqrt(frequency, task)
        gen_log(frequency, task)

    os.system("rm -f *.vhdl")
    os.system("rm -f *.out")

# Post-process: add /* verilator lint_off CASEOVERLAP*/ to all generated verilog files
for dir in os.listdir('.'):
    if not os.path.isdir(dir):
        continue
    for file in os.listdir(dir):
        if not file.endswith('.v'):
            continue
        with open(os.path.join(dir, file), 'r') as f:
            content = f.read()
        content = content.replace('module ', '/* verilator lint_off CASEOVERLAP*/\nmodule ')
        content = content.replace('endmodule', 'endmodule\n/* verilator lint_on CASEOVERLAP*/')
        with open(os.path.join(dir, file), 'w') as f:
            f.write(content)
