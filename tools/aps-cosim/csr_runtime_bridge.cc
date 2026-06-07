#include "csr_runtime_bridge.h"

#include "Vmain.h"
#include "Vmain___024root.h"
#include "Vmain__Syms.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <verilated_sym_props.h>

void CsrRuntimeBridge::drive(Vmain &top, ApsCsrFile &csrs) {
  bind(top, csrs);
  for (const Handles &csr : handles_) {
    putU32(csr.value, csrs.readByName(csr.name));
    putU8(csr.valueReady, 1);
    putU8(csr.setReady, 1);
    if (std::getenv("APS_COSIM_TRACE"))
      std::cerr << "csr " << csr.name << " value=0x" << std::hex
                << getU32(csr.value) << std::dec
                << " value_ready=" << unsigned(getU8(csr.valueReady))
                << " set_ready=" << unsigned(getU8(csr.setReady)) << "\n";
  }
}

void CsrRuntimeBridge::sample(Vmain &top, ApsCsrFile &csrs) {
  bind(top, csrs);
  for (const Handles &csr : handles_) {
    if (getU8(csr.setEnable) != 0)
      csrs.writeByName(csr.name, getU32(csr.setData));
  }
}

void CsrRuntimeBridge::bind(Vmain &top, ApsCsrFile &csrs) {
  if (bound_)
    return;

  for (const auto &entry : csrs.entries()) {
    if (!entry.hasAddress)
      continue;
    Handles handles;
    handles.name = entry.name;
    std::string prefix = "csr_" + entry.name;
    handles.value = requiredPort(top, prefix + "_value_res0");
    handles.valueReady = requiredPort(top, prefix + "_value_ready");
    handles.setData =
        requiredPort(top, prefix + "_set_" + entry.name + "_sdata");
    handles.setReady = requiredPort(top, prefix + "_set_ready");
    handles.setEnable = requiredPort(top, prefix + "_set_enable");
    handles_.push_back(handles);
  }

  bound_ = true;
}

void *CsrRuntimeBridge::requiredPort(Vmain &top, const std::string &name) {
  const VerilatedVar *var =
      top.rootp->vlSymsp->__Vscopep_TOP->varFind(name.c_str());
  if (!var)
    throw std::runtime_error("missing RTL CSR port: " + name);
  return var->datap();
}

void CsrRuntimeBridge::putU32(void *port, uint32_t value) {
  *static_cast<uint32_t *>(port) = value;
}

void CsrRuntimeBridge::putU8(void *port, uint8_t value) {
  *static_cast<uint8_t *>(port) = value;
}

uint32_t CsrRuntimeBridge::getU32(void *port) {
  return *static_cast<uint32_t *>(port);
}

uint8_t CsrRuntimeBridge::getU8(void *port) {
  return *static_cast<uint8_t *>(port);
}
