/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_NUI_RECORDER_H_
#define XENIA_NUI_NUI_RECORDER_H_

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "xenia/base/threading.h"
#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {

class NuiSystem;

// .nuirec: a recording of what the emulated sensor published, one frame per
// 30 Hz tick, playable with nui_source=playback.
// The file format (NuirecWriter/NuirecReader, mask RLE) is implemented in
// nuirec_file.cc; the live NuiRecorder sink is in nui_recorder.cc.
//
// Layout (all integers and floats little-endian):
//   header   char magic[4] = "XNUI"
//            u32  version = 1
//            u32  flags   (kNuirecFlagMask | kNuirecFlagDepth |
//                          kNuirecFlagColor: which sections every frame has)
//            u32  fps     (30)
//            f32  camera_height_m
//            f32  hfov    (degrees; 0 = unknown)
//   frame    u64  timestamp_us
//            u32  body_count (<= kMaxSkeletons)
//            body[body_count]: u32 state, u32 tracking_id (source person key),
//                              u32 enrollment_index, u32 user_index,
//                              f32[4] position, f32[4] joints[20],
//                              u32 joint_states[20], u32 quality_flags
//            f32[4] floor_clip_plane, f32[4] normal_to_gravity
//            if flags & kNuirecFlagMask:
//              u32 run_count, then run_count x {u8 value, u16 length}
//              covering exactly kDepthWidth * kDepthHeight pixels
//            if flags & kNuirecFlagDepth:
//              u32 size, zstd frame of kDepthWidth * kDepthHeight u16 (mm)
//            if flags & kNuirecFlagColor:
//              u32 size, zstd frame of kColorWidth * kColorHeight x RGB u8
//   until end of file.

constexpr uint32_t kNuirecVersion = 1;
constexpr uint32_t kNuirecFlagMask = 1u << 0;
constexpr uint32_t kNuirecFlagDepth = 1u << 1;
constexpr uint32_t kNuirecFlagColor = 1u << 2;
constexpr size_t kNuirecHeaderSize = 4 + 4 + 4 + 4 + 4 + 4;

struct NuirecHeader {
  uint32_t version = kNuirecVersion;
  uint32_t flags = 0;
  uint32_t fps = kFrameRateHz;
  float camera_height_m = 0.0f;
  float hfov_degrees = 0.0f;

  bool has_mask() const { return (flags & kNuirecFlagMask) != 0; }
  bool has_depth() const { return (flags & kNuirecFlagDepth) != 0; }
  bool has_color() const { return (flags & kNuirecFlagColor) != 0; }
};

// One recorded frame in memory. The optional planes are only meaningful when
// the corresponding has_* member is set; on write, a section the header
// declares but the frame lacks is stored as zeros.
struct NuirecFrame {
  int64_t timestamp_us = 0;
  uint32_t body_count = 0;
  std::array<Skeleton, kMaxSkeletons> bodies{};
  Vec4 floor_clip_plane;
  Vec4 normal_to_gravity;
  bool has_player_mask = false;
  std::vector<uint8_t> player_mask;  // kDepthWidth * kDepthHeight
  bool has_depth = false;
  std::vector<uint16_t> depth_mm;  // kDepthWidth * kDepthHeight
  bool has_color = false;
  std::vector<uint8_t> color_rgb;  // kColorWidth * kColorHeight * 3

  void Reset();
};

// Run-length codec for the player mask (shared with the tests).
// Encodes into |out| as u32 run_count followed by {u8 value, u16 length}
// runs; |pixel_count| pixels are consumed from |mask|.
void NuirecEncodeMask(const uint8_t* mask, size_t pixel_count,
                      std::vector<uint8_t>* out);
// Decodes |size| bytes at |data| into |out_mask| (exactly |pixel_count|
// pixels). Returns the number of bytes consumed, or 0 with |out_error| set.
size_t NuirecDecodeMask(const uint8_t* data, size_t size, size_t pixel_count,
                        uint8_t* out_mask, std::string* out_error);

class NuirecWriter {
 public:
  NuirecWriter() = default;
  ~NuirecWriter();
  NuirecWriter(const NuirecWriter&) = delete;
  NuirecWriter& operator=(const NuirecWriter&) = delete;

  // Creates (truncates) |path| and writes the header. Sets |out_error| and
  // returns false on failure.
  bool Open(const std::filesystem::path& path, const NuirecHeader& header,
            std::string* out_error);
  bool WriteFrame(const NuirecFrame& frame, std::string* out_error);
  void Close();

  bool is_open() const { return file_ != nullptr; }
  const NuirecHeader& header() const { return header_; }
  const std::filesystem::path& path() const { return path_; }
  uint64_t frames_written() const { return frames_written_; }

 private:
  FILE* file_ = nullptr;
  std::filesystem::path path_;
  NuirecHeader header_;
  uint64_t frames_written_ = 0;
  std::vector<uint8_t> buffer_;
  std::vector<uint8_t> zero_scratch_;
  std::vector<uint8_t> compress_scratch_;
};

class NuirecReader {
 public:
  NuirecReader() = default;
  ~NuirecReader();
  NuirecReader(const NuirecReader&) = delete;
  NuirecReader& operator=(const NuirecReader&) = delete;

  bool Open(const std::filesystem::path& path, std::string* out_error);
  // Reads the next frame. Returns false at the end of the recording with
  // |out_error| empty, or on a malformed file with |out_error| set.
  bool ReadFrame(NuirecFrame* out_frame, std::string* out_error);
  // Seeks back to the first frame.
  bool Rewind(std::string* out_error);
  void Close();

  bool is_open() const { return file_ != nullptr; }
  const NuirecHeader& header() const { return header_; }
  const std::filesystem::path& path() const { return path_; }
  // Frames returned since Open/Rewind.
  uint64_t frames_read() const { return frames_read_; }

 private:
  bool ReadExact(void* out, size_t size, bool* out_eof, std::string* out_error);

  FILE* file_ = nullptr;
  std::filesystem::path path_;
  NuirecHeader header_;
  uint64_t frames_read_ = 0;
  std::vector<uint8_t> buffer_;
};

// Records every frame the NuiSystem publishes to cvars::nui_record_path.
//
// The recorder attaches a frame sink to the system; the sink (pacer thread)
// copies the frame into a 64-entry ring which a low-priority writer thread
// drains into a NuirecWriter. The cvar is re-read once per second from the
// sink: setting it starts a new file (an existing file is overwritten),
// clearing it closes the current one. The file's sections (mask / depth /
// colour) are decided from the first frame written to it.
//
// Lifecycle: Emulator::Setup calls Get()->Start(nui_system) after
// NuiSystem::Setup; Stop() must be called before the NuiSystem is destroyed.
class NuiRecorder {
 public:
  static NuiRecorder* Get();

  NuiRecorder();
  ~NuiRecorder();
  NuiRecorder(const NuiRecorder&) = delete;
  NuiRecorder& operator=(const NuiRecorder&) = delete;

  // Attaches to |system| (no-op if null or NUI is disabled) and starts the
  // writer thread. Safe to call more than once.
  void Start(NuiSystem* system);
  // Detaches, flushes pending frames, closes the file and joins the thread.
  void Stop();

  // Whether a file is currently open for writing.
  bool is_recording() const;
  // Path of the file being written (empty when not recording).
  std::filesystem::path current_path() const;
  uint64_t frames_written() const {
    return frames_written_.load(std::memory_order_relaxed);
  }
  uint64_t frames_dropped() const {
    return frames_dropped_.load(std::memory_order_relaxed);
  }
  std::string last_error() const;

 private:
  static constexpr size_t kRingSize = 64;

  struct Entry {
    enum class Kind { kFrame, kClose };
    Kind kind = Kind::kFrame;
    std::filesystem::path path;
    NuirecFrame frame;
  };

  void OnFrame(uint32_t frame_number, int64_t timestamp_us,
               const SkeletonFrame& skeleton,
               const std::shared_ptr<const SourceFrame>& source);
  void PollRecordPath();
  void WriterThreadMain();
  void ProcessEntry(Entry* entry);
  bool OpenFileFor(const Entry& entry);
  void CloseFile();
  void SetError(const std::string& error);

  NuiSystem* system_ = nullptr;
  uint32_t sink_id_ = 0;

  // Pacer-thread state.
  std::filesystem::path active_path_;
  std::chrono::steady_clock::time_point last_poll_;

  // Ring between the sink and the writer thread.
  std::mutex ring_mutex_;
  std::condition_variable ring_cv_;
  std::array<Entry, kRingSize> ring_;
  size_t ring_head_ = 0;
  size_t ring_count_ = 0;
  bool stop_ = false;
  std::unique_ptr<xe::threading::Thread> thread_;

  // Writer-thread state.
  Entry scratch_;
  NuirecWriter writer_;
  std::filesystem::path failed_path_;

  std::atomic<uint64_t> frames_written_{0};
  std::atomic<uint64_t> frames_dropped_{0};
  mutable std::mutex status_mutex_;
  std::filesystem::path open_path_;
  std::string last_error_;
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_NUI_RECORDER_H_
