#pragma once
// ── ModelDrift Sabotage Defense — POSIX Shared Memory Ring Buffer ──────────────────────────
// Binary-safe lock-free ring buffer for inter-process weight exchange.
// No JSON. Uses raw float arrays + atomic sequence numbers.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>

namespace fcmdr {

static constexpr size_t SHM_CAPACITY  = 64;       // ring slots
static constexpr size_t WEIGHTS_PER_SLOT = 256;   // max floats per slot

struct WeightSlot {
  std::atomic<uint64_t> seq;          // sequence number (even = writable)
  float   weights[WEIGHTS_PER_SLOT];
  float   loss;
  uint8_t node_id[32];
  uint64_t ts_ns;
};

struct ShmRingBuffer {
  std::atomic<uint64_t> head;
  std::atomic<uint64_t> tail;
  WeightSlot slots[SHM_CAPACITY];
};

class ShmWriter {
public:
  explicit ShmWriter(const std::string& name)
    : name_("/dev/shm/" + name), fd_(-1), buf_(nullptr)
  {
    fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd_ < 0) throw std::runtime_error("shm_open failed");
    ftruncate(fd_, sizeof(ShmRingBuffer));
    buf_ = static_cast<ShmRingBuffer*>(
      mmap(nullptr, sizeof(ShmRingBuffer),
           PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0));
    if (buf_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    new (buf_) ShmRingBuffer();
  }

  ~ShmWriter() {
    if (buf_) munmap(buf_, sizeof(ShmRingBuffer));
    if (fd_ >= 0) close(fd_);
  }

  bool write(const float* weights, size_t n, float loss,
             const char* node_id, uint64_t ts_ns) {
    uint64_t tail = buf_->tail.fetch_add(1, std::memory_order_relaxed);
    size_t   idx  = tail % SHM_CAPACITY;
    WeightSlot& s = buf_->slots[idx];

    uint64_t expected = tail * 2;
    while (s.seq.load(std::memory_order_acquire) != expected) {
      // spin-wait — slot still being read
    }

    size_t copy_n = n < WEIGHTS_PER_SLOT ? n : WEIGHTS_PER_SLOT;
    std::memcpy(s.weights, weights, copy_n * sizeof(float));
    s.loss = loss;
    std::strncpy(reinterpret_cast<char*>(s.node_id), node_id, 31);
    s.ts_ns = ts_ns;

    s.seq.store(tail * 2 + 1, std::memory_order_release);
    return true;
  }

private:
  std::string    name_;
  int            fd_;
  ShmRingBuffer* buf_;
};

class ShmReader {
public:
  explicit ShmReader(const std::string& name)
    : name_("/dev/shm/" + name), fd_(-1), buf_(nullptr)
  {
    fd_ = shm_open(name_.c_str(), O_RDONLY, 0600);
    if (fd_ < 0) throw std::runtime_error("shm_open failed");
    buf_ = static_cast<ShmRingBuffer*>(
      mmap(nullptr, sizeof(ShmRingBuffer),
           PROT_READ, MAP_SHARED, fd_, 0));
    if (buf_ == MAP_FAILED) throw std::runtime_error("mmap failed");
  }

  ~ShmReader() {
    if (buf_) munmap(buf_, sizeof(ShmRingBuffer));
    if (fd_ >= 0) close(fd_);
  }

  bool try_read(float* out_weights, size_t n, float& loss,
                std::string& node_id, uint64_t& ts_ns) {
    uint64_t head = buf_->head.load(std::memory_order_relaxed);
    size_t   idx  = head % SHM_CAPACITY;
    WeightSlot& s = buf_->slots[idx];

    uint64_t seq = s.seq.load(std::memory_order_acquire);
    if ((seq & 1) == 0) return false;  // not ready

    size_t copy_n = n < WEIGHTS_PER_SLOT ? n : WEIGHTS_PER_SLOT;
    std::memcpy(out_weights, s.weights, copy_n * sizeof(float));
    loss    = s.loss;
    node_id = reinterpret_cast<char*>(s.node_id);
    ts_ns   = s.ts_ns;

    if (s.seq.load(std::memory_order_acquire) != seq)
      return false;  // torn read

    buf_->head.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

private:
  std::string    name_;
  int            fd_;
  ShmRingBuffer* buf_;
};

} // namespace fcmdr
