#ifndef APS_COSIM_FAKE_DMA_H
#define APS_COSIM_FAKE_DMA_H

#include "memory_bridge.h"

#include <cstdint>
#include <deque>

class Vmain;

class FakeDma {
public:
  explicit FakeDma(MemoryBridge &memory);

  void driveInputs(Vmain &top);
  void sampleOutputs(Vmain &top);
  bool idle() const;

private:
  enum class Kind { CpuToIsax, IsaxToCpu };

  struct Request {
    Kind kind;
    unsigned channel;
    uint32_t cpuAddr;
    uint32_t isaxAddr;
    uint32_t bytes;
    uint8_t strideX;
    uint8_t strideY;
  };

  enum class State {
    Idle,
    BurstWrite,
    BurstReadAddr,
    BurstReadData
  };

  static uint32_t bytesFromLength(uint8_t length);
  void enqueueDmaRequests(const Vmain &top);
  void serviceHella(const Vmain &top);
  void advanceTransfer();

  MemoryBridge &memory_;
  State state_ = State::Idle;
  std::deque<Request> queue_;
  Request current_{Kind::CpuToIsax, 0, 0, 0, 0, 0, 0};
  uint32_t offset_ = 0;
  uint32_t currentIsaxAddr_ = 0;
  uint8_t tileOffset_ = 0;
  uint8_t hellaRespTag_ = 0;
  uint8_t hellaRespCmd_ = 0;
  uint8_t hellaRespSize_ = 0;
  uint32_t hellaRespData_ = 0;
  bool hellaRespValid_ = false;
};

#endif
