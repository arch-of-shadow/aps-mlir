#!/bin/bash
# Convert VHDL to Verilog using yosys + ghdl
#
# 目的: FloPoCo 生成 VHDL，但 CIRCT/aps-mlir 需要 Verilog
# 方法: 使用 yosys 的 ghdl 插件进行转换

set -e

for vhdl_file in *.vhdl; do
    if [ ! -f "$vhdl_file" ]; then
        continue
    fi

    base_name="${vhdl_file%.vhdl}"
    v_file="${base_name}.v"

    # 从 VHDL 文件中提取顶层实体名（最后一个 entity 声明）
    top_entity=$(grep -i "^entity" "$vhdl_file" | tail -1 | awk '{print $2}')

    echo "Converting $vhdl_file (top entity: $top_entity) -> $v_file"

    # 使用 yosys + ghdl 插件转换:
    # 1. ghdl: 读取 VHDL 并 elaborate 顶层实体
    # 2. hierarchy: 设置顶层模块
    # 3. rename: 将实体名从 FixSinCos_24 重命名为带延迟后缀的名称
    # 4. write_verilog: 输出 Verilog
    yosys -m ghdl -p "
        ghdl --std=08 -fsynopsys -fexplicit ${vhdl_file} -e ${top_entity};
        hierarchy -top ${top_entity};
        rename ${top_entity} ${base_name};
        write_verilog ${v_file}
    " 2>&1 | grep -E "(Executing|Generated|Writing|error)" || true

    # 添加 verilator lint 指令（避免综合警告）
    if [ -f "$v_file" ]; then
        sed -i '1s/^/\/* verilator lint_off CASEOVERLAP *\/\n/' "$v_file"
        echo "  -> Generated: $v_file"
    else
        echo "  -> FAILED: $v_file not created"
    fi
done

rm -f modules.rpt
echo "Done."
