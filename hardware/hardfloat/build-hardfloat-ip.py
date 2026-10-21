#!/usr/bin/env python3
"""Package pinned HardFloat as SV and a Yosys-imported FIRRTL implementation.

Named key=value arguments match the CMT2 ModuleLibrary build convention.
Run in the desired output directory; --resource-output overlays the existing
integer resource database with wrapper timing (not a physical characterization).
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "thirdparty/HardFloat/source"
CATALOG = ROOT / "hardware/hardfloat/ip_catalog.yaml"


def verify_sources():
    vendor = SOURCE.parent
    for line in (vendor / "SOURCE.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((vendor / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"HardFloat source hash mismatch: {name}")


def expand_includes(path):
    def replace(match):
        name = match.group(1)
        candidates = [SOURCE / "RISCV" / name, SOURCE / name]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise ValueError(f"Cannot resolve {name} in {path}")
        return expand_includes(found)

    text = re.sub(r'`include\s+"([^"]+)"', replace, path.read_text())
    # Release 1 redundantly redeclares an ANSI output net. Normalize for SV,
    # retaining the unmodified, hash-verified upstream sources.
    if path.name == 'divSqrtRecFN_small.v':
        text = text.replace('    wire sqrtOpOut;\n', '')
    # Release 1 uses ceil(log2(p)), insufficient when p is a power of two:
    # BF16 p=8 saturates alignment at 7, inside the guard/round window.
    # Reserve enough range for p+3; unchanged for binary32 p=24 (5 bits).
    if path.name == 'addRecFN.v':
        text = text.replace('localparam alignDistWidth = clog2(sigWidth);',
                            'localparam alignDistWidth = clog2(sigWidth + 3);')
    return text


def atomic_write(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def bound_shift_widths(module):
    """Keep truncated Verilog shifts from creating exponential FIRRTL widths."""
    next_bit = 1 + max(bit for net in module["netnames"].values()
                       for bit in net["bits"] if isinstance(bit, int))
    changed = False
    for index, cell in enumerate(list(module["cells"].values())):
        if cell["type"] not in ("$shl", "$sshl"):
            continue
        params = cell["parameters"]
        width = int(params["Y_WIDTH"], 2)
        amount_width = int(params["B_WIDTH"], 2)
        useful = max(1, (width - 1).bit_length())
        if int(params["B_SIGNED"], 2) or amount_width <= useful:
            continue
        value_name, large_name = f"hf_shift_value_{index}", f"hf_shift_large_{index}"
        if any(name in module["netnames"] or name in module["cells"]
               for name in (value_name, large_name, value_name + "_cell",
                            large_name + "_cell")):
            raise ValueError("HardFloat shift import name collision")
        value = list(range(next_bit, next_bit + width))
        large = [next_bit + width]
        next_bit += width + 1
        amount = cell["connections"]["B"]
        output = cell["connections"]["Y"]
        cell["connections"]["B"] = amount[:useful]
        cell["connections"]["Y"] = value
        params["B_WIDTH"] = f"{useful:032b}"
        for name, bits in ((value_name, value), (large_name, large)):
            module["netnames"][name] = dict(hide_name=0, bits=bits, attributes={})
        # Any nonzero upper shift bit implies a distance >= the output width,
        # whose truncated left-shift result is zero. Lower distances retain
        # the original shift, including its signed data extension semantics.
        module["cells"][large_name + "_cell"] = dict(
            hide_name=0, type="$reduce_or", attributes={},
            parameters=dict(A_SIGNED="0", A_WIDTH=f"{amount_width-useful:032b}",
                            Y_WIDTH=f"{1:032b}"),
            port_directions=dict(A="input", Y="output"),
            connections=dict(A=amount[useful:], Y=large))
        module["cells"][value_name + "_cell"] = dict(
            hide_name=0, type="$mux", attributes={},
            parameters=dict(WIDTH=f"{width:032b}"),
            port_directions=dict(A="input", B="input", S="input", Y="output"),
            connections=dict(A=value, B=["0"] * width, S=large, Y=output))
        changed = True
    return changed


def emit_firrtl(name, bundle, specialization):
    """Import a flattened implementation while preserving the public port ABI."""
    yosys = os.environ.get("YOSYS", shutil.which("yosys") or
                           str(ROOT / ".pixi/envs/default/bin/yosys"))
    firtool = os.environ.get("FIRTOOL", str(ROOT / "circt/build/bin/firtool"))
    with tempfile.TemporaryDirectory(prefix=name + ".", dir=Path.cwd()) as work:
        work = Path(work)
        firrtl = work / "implementation.fir"
        netlist = work / "implementation.json"
        output = work / "implementation.mlir"
        script = (
            f"read_verilog -sv -DSYNTHESIS {json.dumps(str(bundle))} "
            f"{json.dumps(str(specialization))}; "
            f"hierarchy -check -top {name}; "
            "proc; flatten; opt; memory_map; opt; dffunmap; "
            "setundef -undriven -zero; splitnets -driver; opt_clean; "
            f"check -assert; write_json {json.dumps(str(netlist))}"
        )
        # Map response queues to registers and unmap FF enables/resets to the
        # cells supported by write_firrtl. Split only at driver boundaries:
        # partial vector assignments can otherwise form false FIRRTL cycles.
        result = subprocess.run([yosys, "-Q", "-T", "-p", script],
                                capture_output=True, text=True)
        if result.returncode:
            raise ValueError("HardFloat Yosys import failed:\n" +
                             (result.stdout + result.stderr)[-6000:])
        # write_firrtl supports $dff but not $adff. Carry async reset metadata
        # separately, export the register, then attach its original reset in
        # FIRRTL. Do not use async2sync: reset behavior must be preserved.
        design = json.loads(netlist.read_text())
        module = design["modules"][name]
        bounded_shifts = bound_shift_widths(module)
        async_resets = []
        for cell_name, cell in list(module["cells"].items()):
            if cell["type"] != "$adff":
                continue
            index = len(async_resets)
            register = f"hf_async_reg_{index}"
            reset = f"hf_async_reset_{index}"
            if register in module["cells"] or reset in module["netnames"]:
                raise ValueError("HardFloat async-reset import name collision")
            params = cell["parameters"]
            width = int(params["WIDTH"], 2)
            polarity = int(params.pop("ARST_POLARITY"), 2)
            value = int(params.pop("ARST_VALUE"), 2)
            module["netnames"][reset] = dict(
                hide_name=0, bits=cell["connections"].pop("ARST"), attributes={})
            cell["port_directions"].pop("ARST")
            cell["type"] = "$dff"
            del module["cells"][cell_name]
            module["cells"][register] = cell
            async_resets.append((register, reset, width, polarity, value))
        netlist.write_text(json.dumps(design))
        # Iterative cores also feed individual bits of packed flag vectors
        # back into other bits. Split those internal nets fully so FIRRTL's
        # vector-level cycle checker sees the actual bit dependencies.
        if async_resets or bounded_shifts:
            script = (f"read_json {json.dumps(str(netlist))}; splitnets; "
                      f"write_firrtl {json.dumps(str(firrtl))}")
        else:
            # Preserve Yosys's original alias ordering for the fixed cores.
            # JSON reimport can aggregate unrelated vector bits into aliases.
            script = script.rsplit("write_json ", 1)[0] + (
                f"write_firrtl {json.dumps(str(firrtl))}")
        result = subprocess.run([yosys, "-Q", "-T", "-p", script],
                                capture_output=True, text=True)
        if result.returncode:
            raise ValueError("HardFloat FIRRTL export failed:\n" +
                             (result.stdout + result.stderr)[-6000:])
        text = re.sub(r" @\[[^\n]*?\]", "", firrtl.read_text())
        for register, reset, width, polarity, value in async_resets:
            predicate = reset if polarity else f"not({reset})"
            reset_clause = (f" with : (reset => (asAsyncReset({predicate}), "
                            f"UInt<{width}>({value})))")
            text, count = re.subn(
                rf"^(\s*reg {register}: UInt<{width}>, [^\n]+)$",
                lambda match: match[1] + reset_clause, text, flags=re.M)
            if count != 1:
                raise ValueError(f"Missing imported async register: {register}")
        # Verilog has no Clock type. Expose the existing CMT2 clock ABI and
        # retain UInt uses inside the imported implementation via asUInt.
        header, body = text.split("\n\n", 1)
        old_clock = "    input clock: UInt<1>"
        if header.count(old_clock) != 1:
            raise ValueError("HardFloat import requires one UInt<1> clock input")
        header = header.replace(old_clock, "    input clock: Clock")
        body = re.sub(r"\bclock\b", "clock_bits", body)
        firrtl.write_text("FIRRTL version 2.0.0\n" + header +
                          "\n\n    node clock_bits = asUInt(clock)\n" + body)
        result = subprocess.run(
            [firtool, str(firrtl), "--ir-fir", "--disable-all-randomization",
             "--strip-debug-info", "-o", str(output)],
            capture_output=True, text=True)
        if result.returncode:
            raise ValueError("HardFloat FIRRTL import failed:\n" +
                             (result.stdout + result.stderr)[-6000:])
        atomic_write(Path(name + ".mlir"), output.read_text())


def native_timing(width, operation):
    catalog = yaml.safe_load(CATALOG.read_text())
    spec = list(catalog['operations'].values())[operation]
    variable = 'sig_latency' in spec
    latency = width - 8 + spec['sig_latency'] if variable else spec['latency']
    return dict(variable_latency=variable, latency_upper_bound=latency,
                no_stall_II=latency if variable else 1,
                outstanding_capacity=1 if variable else latency + catalog['response_slack'])


def transaction_timing(width, operation, latency, capacity):
    timing = native_timing(width, operation)
    if latency != timing['latency_upper_bound']:
        raise ValueError(f"Stale HardFloat latency {latency}; native contract requires {timing['latency_upper_bound']}")
    if timing['variable_latency'] and capacity != 1:
        raise ValueError("Iterative HardFloat capacity must be 1")
    timing['outstanding_capacity'] = capacity
    return timing


def main():
    catalog = yaml.safe_load(CATALOG.read_text())
    params = {"op": 0, "latency": None, "capacity": None, "width": 0, "pred": 0}
    resource_output = None
    cpp_output = None
    for arg in sys.argv[1:]:
        if arg.startswith("--cpp-output="):
            cpp_output = Path(arg.split("=", 1)[1])
        elif arg.startswith("--resource-output="):
            resource_output = Path(arg.split("=", 1)[1])
        else:
            key, value = arg.split("=", 1)
            if key not in params:
                raise ValueError(f"Unknown parameter: {key}")
            params[key] = int(value)
    if params['op'] not in range(19) or params['width'] not in (0,16,32):
        raise ValueError("Invalid operation or precision")
    timing = native_timing(params['width'] or 32, params['op'])
    if params['latency'] is None: params['latency'] = timing['latency_upper_bound']
    if params['capacity'] is None: params['capacity'] = timing['outstanding_capacity']
    op, latency, capacity = (params[k] for k in ("op", "latency", "capacity"))
    universal = params["width"] != 0
    if op not in range(19 if universal else 3) or not 1 <= capacity <= 256:
        raise ValueError("Invalid operation or capacity")
    transaction_timing(params['width'] or 32, op, latency, capacity)
    if cpp_output:
        cpp_output.parent.mkdir(parents=True, exist_ok=True)
        rows = ',\n'.join('  {' + ','.join(str(native_timing(w, k)['latency_upper_bound']) for k in range(19)) + '}' for w in (16,32))
        cpp_output.write_text(f'''// Generated from hardware/hardfloat/ip_catalog.yaml; do not edit.
#ifndef APS_HARDFLOATCONFIG_H
#define APS_HARDFLOATCONFIG_H
namespace mlir::hardfloat {{
constexpr unsigned latencies[2][19] = {{
{rows}
}};
constexpr unsigned latency(unsigned op, unsigned width) {{ return latencies[width==16 ? 0 : 1][op]; }}
constexpr unsigned capacity(unsigned op, unsigned width) {{ return op==3 || op==4 ? 1 : latency(op,width)+{catalog['response_slack']}; }}
}} // namespace mlir::hardfloat
#endif
''')
        return
    verify_sources()

    # Both standalone RTL tests and FIRRTL import use this source bundle.
    # No external include paths remain in the bundle.
    bundle = ROOT / "build/hardfloat/rtl/HardFloatIP.sv"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    dependencies = [
        "HardFloat_primitives.v",
        "RISCV/HardFloat_specialize.v",
        "HardFloat_rawFN.v",
        "isSigNaNRecFN.v",
        "fNToRecFN.v",
        "recFNToFN.v",
        "addRecFN.v",
        "mulRecFN.v", "mulAddRecFN.v", "compareRecFN.v", "iNToRecFN.v",
        "recFNToIN.v", "recFNToRecFN.v", "divSqrtRecFN_small.v",
    ]
    contents = "// Generated from pinned HardFloat Release 1; see vendor license.\n"
    expressions=[]
    for kind, spec in enumerate(catalog['operations'].values()):
        value=f"((W)-8+{spec['sig_latency']})" if 'sig_latency' in spec else str(spec['latency'])
        expressions.append(f"(O)=={kind} ? {value} : ")
    contents += "`define HF_NATIVE_LATENCY(W,O) (" + ''.join(expressions) + "0)\n"
    contents += f"`define HF_NATIVE_CAPACITY(W,O) (((O)==3 || (O)==4) ? 1 : (`HF_NATIVE_LATENCY(W,O)+{catalog['response_slack']}))\n"
    contents += "\n".join(expand_includes(SOURCE / name) for name in dependencies)
    for name in ("HF32AddSub", "HF32Mul", "HF32Binary", "HFCompute", "HFNativePipeline", "HFUnit"):
        contents += "\n" + (ROOT / f"hardware/hardfloat/{name}.sv").read_text()
    # Avoid unnecessary rewrites of the shared bundle (parallel build readers).
    if not bundle.exists() or bundle.read_text() != contents:
        temp = bundle.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(contents)
        temp.replace(bundle)

    if universal:
        emit_universal(params, catalog, bundle)
        return
    name = f"HardFloat32_op{op}_l{latency}_q{capacity}"
    # Specialize before importing so parameters become concrete FIRRTL logic.
    specialization = bundle.parent / f"{name}.sv"
    specialization.write_text(
        f"""module {name} (
    input wire clock, reset, issue_enable,
    output wire issue_ready,
    input wire [31:0] operand0, operand1,
    input wire collect_enable,
    output wire collect_ready,
    output wire [31:0] collect_data,
    output wire [4:0] collect_flags
);
    HF32Binary #(.OP({op}), .LATENCY({latency}), .CAPACITY({capacity})) impl (.*);
endmodule
"""
    )
    emit_firrtl(name, bundle, specialization)
    Path(f"{name}.f").write_text(str(bundle) + "\n" + str(specialization) + "\n")
    metadata = dict(catalog, **params, module=name, rtl=str(bundle))
    Path(f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if resource_output:
        resources = json.loads((ROOT / "examples/resource_ihp130.json").read_text())
        for kind, spec in enumerate(catalog['operations'].values()):
            resource_name = spec['resource']
            for width, suffix in ((16, '_bf16'), (32, '_fp32')):
                t = native_timing(width, kind)
                resources[resource_name + suffix] = dict(delay=[0]*7, latency=[t['latency_upper_bound']]*7,
                    II=t['no_stall_II'], constr=0, amount=-1)
            resources[resource_name] = dict(resources[resource_name + '_fp32'])
            resources[resource_name]['latency'] = list(resources[resource_name]['latency'])
            resources[resource_name]['latency'][4] = native_timing(16, kind)['latency_upper_bound']
        resources['fp'] = dict(resources['addf'])
        resource_output.parent.mkdir(parents=True, exist_ok=True)
        resource_output.write_text(json.dumps(resources, indent=2) + "\n")
    print(f"{name}.mlir")


def emit_universal(params, catalog, bundle):
    w,op,pred,l,q = (params[k] for k in ('width','op','pred','latency','capacity'))
    if w not in (16,32) or pred not in range(16):
        raise ValueError('Require width=16 (BF16)/32 (FP32), pred=0..15')
    name = f'HardFloat_w{w}_o{op}_p{pred}_l{l}_q{q}'
    impl = bundle.parent / (name+'.sv')
    atomic_write(impl, f'''module {name}(
 input wire clock,reset,issue_enable, output wire issue_ready,
 input wire [31:0] operand0,operand1,operand2, input wire [2:0] rounding_mode,
 input wire collect_enable, output wire collect_ready,
 output wire [31:0] collect_data, output wire [4:0] collect_flags
);
 HFUnit #(.WIDTH({w}),.OP({op}),.PRED({pred}),.LATENCY({l}),.CAPACITY({q})) impl(.*);
endmodule
''')
    emit_firrtl(name, bundle, impl)
    atomic_write(Path(name+'.f'), str(bundle)+'\n'+str(impl)+'\n')
    timing = transaction_timing(w, op, l, q)
    metadata = dict(catalog, **params)
    metadata.update(II=timing['no_stall_II'], timing=timing)
    metadata['latency_parameter_meaning'] = 'transaction_upper_bound' if op in (3, 4) else 'transaction'
    atomic_write(Path(name+'.json'), json.dumps(metadata,indent=2)+'\n')
    print(name+'.mlir')


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
