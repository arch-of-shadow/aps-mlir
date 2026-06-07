#include "fake_dma.h"

#include "Vmain.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

FakeDma::FakeDma(MemoryBridge &memory) : memory_(memory) {}

namespace {
bool traceEnabled() { return std::getenv("APS_COSIM_TRACE") != nullptr; }
}

bool FakeDma::idle() const {
  return state_ == State::Idle && queue_.empty();
}

uint32_t FakeDma::bytesFromLength(uint8_t length) {
  if (length >= 31)
    throw std::runtime_error("DMA length encoding is too large");
  return 1u << length;
}

void FakeDma::driveInputs(Vmain &top) {
  const bool canQueue = queue_.size() < 16;
  const bool isIdle = idle();
  top.dma_cpu_to_isax_ch0_ready = canQueue;
  top.dma_isax_to_cpu_ch0_ready = canQueue;
  top.dma_poll_for_idle_ch0_ready = isIdle;
  top.dma_poll_for_idle_ch0_res0 = isIdle;
  top.dma_cpu_to_isax_ch1_ready = canQueue;
  top.dma_isax_to_cpu_ch1_ready = canQueue;
  top.dma_poll_for_idle_ch1_ready = isIdle;
  top.dma_poll_for_idle_ch1_res0 = isIdle;

  top.burst_read_0_enable = 0;
  top.burst_read_0_addr = 0;
  top.burst_read_1_enable = 0;
  top.burst_write_enable = 0;
  top.burst_write_addr = 0;
  top.burst_write_data = 0;

  top.hella_cmd_hella_cmd_to_bus_ready = !hellaRespValid_;
  top.hella_resp_enable = hellaRespValid_;
  top.hella_resp_hella_resp_data = hellaRespValid_ ? hellaRespData_ : 0;
  top.hella_resp_hella_resp_tag = hellaRespValid_ ? hellaRespTag_ : 0;
  top.hella_resp_hella_resp_cmd = hellaRespValid_ ? hellaRespCmd_ : 0;
  top.hella_resp_hella_resp_size = hellaRespValid_ ? hellaRespSize_ : 0;
  top.hella_resp_hella_resp_signed = 0;

  if (state_ == State::Idle && !queue_.empty()) {
    current_ = queue_.front();
    queue_.pop_front();
    offset_ = 0;
    currentIsaxAddr_ = current_.isaxAddr;
    tileOffset_ = 0;
    state_ = current_.kind == Kind::CpuToIsax ? State::BurstWrite
                                              : State::BurstReadAddr;
  }

  switch (state_) {
  case State::Idle:
    break;
  case State::BurstWrite: {
    uint32_t bytes = std::min<uint32_t>(8, current_.bytes - offset_);
    uint64_t data = memory_.load64(current_.cpuAddr + offset_);
    top.burst_write_enable = 1;
    top.burst_write_addr = currentIsaxAddr_;
    top.burst_write_data = data;
    (void)bytes;
    break;
  }
  case State::BurstReadAddr:
    top.burst_read_0_enable = 1;
    top.burst_read_0_addr = currentIsaxAddr_;
    break;
  case State::BurstReadData:
    top.burst_read_1_enable = 1;
    break;
  }
}

void FakeDma::sampleOutputs(Vmain &top) {
  enqueueDmaRequests(top);

  if (hellaRespValid_ && top.hella_resp_ready)
    hellaRespValid_ = false;

  serviceHella(top);

  switch (state_) {
  case State::Idle:
    break;
  case State::BurstWrite:
    if (top.burst_write_ready) {
      if (traceEnabled())
        std::cerr << "fake-dma write isax=0x" << std::hex
                  << currentIsaxAddr_ << " cpu=0x"
                  << current_.cpuAddr + offset_ << std::dec << "\n";
      advanceTransfer();
      if (offset_ >= current_.bytes) {
        if (traceEnabled())
          std::cerr << "fake-dma done\n";
        state_ = State::Idle;
      } else {
        state_ = State::BurstWrite;
      }
    }
    break;
  case State::BurstReadAddr:
    if (top.burst_read_0_ready) {
      if (traceEnabled())
        std::cerr << "fake-dma read-addr isax=0x" << std::hex
                  << currentIsaxAddr_ << std::dec << "\n";
      state_ = State::BurstReadData;
    }
    break;
  case State::BurstReadData:
    if (top.burst_read_1_ready) {
      if (traceEnabled())
        std::cerr << "fake-dma read-data cpu=0x" << std::hex
                  << current_.cpuAddr + offset_ << " data=0x"
                  << static_cast<uint64_t>(top.burst_read_1_res0) << std::dec
                  << "\n";
      memory_.store64(current_.cpuAddr + offset_, top.burst_read_1_res0);
      advanceTransfer();
      if (offset_ >= current_.bytes) {
        if (traceEnabled())
          std::cerr << "fake-dma done\n";
        state_ = State::Idle;
      } else {
        state_ = State::BurstReadAddr;
      }
    }
    break;
  }
}

void FakeDma::enqueueDmaRequests(const Vmain &top) {
  auto enqueue = [this](Request req) {
    if (traceEnabled())
      std::cerr << "fake-dma enqueue "
                << (req.kind == Kind::CpuToIsax ? "cpu_to_isax"
                                                 : "isax_to_cpu")
                << " ch" << req.channel << " cpu=0x" << std::hex
                << req.cpuAddr << " isax=0x" << req.isaxAddr << " bytes=0x"
                << req.bytes << " stride_x=0x" << unsigned(req.strideX)
                << " stride_y=0x" << unsigned(req.strideY) << std::dec
                << "\n";
    queue_.push_back(req);
  };

  if (top.dma_cpu_to_isax_ch0_enable && top.dma_cpu_to_isax_ch0_ready)
    enqueue({Kind::CpuToIsax, 0, top.dma_cpu_to_isax_ch0_cpu_addr,
             top.dma_cpu_to_isax_ch0_isax_addr,
             bytesFromLength(top.dma_cpu_to_isax_ch0_length),
             top.dma_cpu_to_isax_ch0_stride_x,
             top.dma_cpu_to_isax_ch0_stride_y});
  if (top.dma_isax_to_cpu_ch0_enable && top.dma_isax_to_cpu_ch0_ready)
    enqueue({Kind::IsaxToCpu, 0, top.dma_isax_to_cpu_ch0_cpu_addr,
             top.dma_isax_to_cpu_ch0_isax_addr,
             bytesFromLength(top.dma_isax_to_cpu_ch0_length),
             top.dma_isax_to_cpu_ch0_stride_x,
             top.dma_isax_to_cpu_ch0_stride_y});
  if (top.dma_cpu_to_isax_ch1_enable && top.dma_cpu_to_isax_ch1_ready)
    enqueue({Kind::CpuToIsax, 1, top.dma_cpu_to_isax_ch1_cpu_addr,
             top.dma_cpu_to_isax_ch1_isax_addr,
             bytesFromLength(top.dma_cpu_to_isax_ch1_length),
             top.dma_cpu_to_isax_ch1_stride_x,
             top.dma_cpu_to_isax_ch1_stride_y});
  if (top.dma_isax_to_cpu_ch1_enable && top.dma_isax_to_cpu_ch1_ready)
    enqueue({Kind::IsaxToCpu, 1, top.dma_isax_to_cpu_ch1_cpu_addr,
             top.dma_isax_to_cpu_ch1_isax_addr,
             bytesFromLength(top.dma_isax_to_cpu_ch1_length),
             top.dma_isax_to_cpu_ch1_stride_x,
             top.dma_isax_to_cpu_ch1_stride_y});
}

void FakeDma::advanceTransfer() {
  offset_ += 8;
  if (current_.strideX != 0 && tileOffset_ + 8 >= current_.strideX) {
    currentIsaxAddr_ +=
        static_cast<uint32_t>(current_.strideX) * current_.strideY -
        tileOffset_;
    tileOffset_ = 0;
  } else {
    currentIsaxAddr_ += 8;
    tileOffset_ += 8;
  }
}

void FakeDma::serviceHella(const Vmain &top) {
  if (!top.hella_cmd_hella_cmd_to_bus_enable ||
      !top.hella_cmd_hella_cmd_to_bus_ready)
    return;

  uint32_t addr = top.hella_cmd_hella_cmd_to_bus_cmd_addr;
  uint8_t cmd = top.hella_cmd_hella_cmd_to_bus_cmd_cmd;
  uint8_t size = top.hella_cmd_hella_cmd_to_bus_cmd_size;
  uint32_t data = top.hella_cmd_hella_cmd_to_bus_cmd_data;
  uint8_t mask = top.hella_cmd_hella_cmd_to_bus_cmd_mask;

  if (traceEnabled())
    std::cerr << "hella cmd=" << unsigned(cmd) << " addr=0x" << std::hex
              << addr << " data=0x" << data << " mask=0x" << unsigned(mask)
              << std::dec << "\n";

  // Rocket M_XRD = 0, M_XWR = 1.
  if (cmd == 0) {
    hellaRespData_ = size == 0 ? memory_.load8(addr) : memory_.load32(addr);
  } else if (cmd == 1) {
    if (size == 0)
      memory_.store8(addr, static_cast<uint8_t>(data));
    else
      memory_.store32(addr, data, mask);
    hellaRespData_ = 0;
  } else {
    throw std::runtime_error("unsupported HellaCache command");
  }

  hellaRespTag_ = top.hella_cmd_hella_cmd_to_bus_cmd_tag;
  hellaRespCmd_ = cmd;
  hellaRespSize_ = size;
  hellaRespValid_ = true;
}
