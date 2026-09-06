/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/nui/nui_recorder.h"

#include <algorithm>
#include <cstring>

#include "third_party/zstd/lib/zstd.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"

// The .nuirec file format (NuirecWriter / NuirecReader and the mask RLE
// helpers). Kept in its own translation unit so tools and tests that only
// read or write files do not link the NuiRecorder (and thus NuiSystem).

namespace xe {
namespace nui {

namespace {

constexpr size_t kMaskPixels = static_cast<size_t>(kDepthWidth) * kDepthHeight;
constexpr size_t kDepthBytes = kMaskPixels * sizeof(uint16_t);
constexpr size_t kColorBytes =
    static_cast<size_t>(kColorWidth) * kColorHeight * 3;
constexpr size_t kSkeletonBytes =
    4 * 4 + 16 + kJointCount * 16 + kJointCount * 4 + 4;
constexpr int kZstdLevel = 1;
// Corrupt sizes are rejected before allocating: nothing compresses worse
// than this.
constexpr uint32_t kMaxCompressedSize = 4u * 1024 * 1024;

// Little-endian serialization helpers. Hosts are little-endian x86-64 /
// arm64, but keep the byte order explicit.

void PutU16(std::vector<uint8_t>* out, uint16_t value) {
  out->push_back(static_cast<uint8_t>(value & 0xFF));
  out->push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

void PutU32(std::vector<uint8_t>* out, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFF));
  }
}

void PutU64(std::vector<uint8_t>* out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFF));
  }
}

void PutF32(std::vector<uint8_t>* out, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  PutU32(out, bits);
}

void PutVec4(std::vector<uint8_t>* out, const Vec4& v) {
  PutF32(out, v.x);
  PutF32(out, v.y);
  PutF32(out, v.z);
  PutF32(out, v.w);
}

uint16_t GetU16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t GetU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t GetU64(const uint8_t* p) {
  return static_cast<uint64_t>(GetU32(p)) |
         (static_cast<uint64_t>(GetU32(p + 4)) << 32);
}

float GetF32(const uint8_t* p) {
  uint32_t bits = GetU32(p);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Vec4 GetVec4(const uint8_t* p) {
  return {GetF32(p), GetF32(p + 4), GetF32(p + 8), GetF32(p + 12)};
}

void PutSkeleton(std::vector<uint8_t>* out, const Skeleton& body) {
  PutU32(out, static_cast<uint32_t>(body.state));
  PutU32(out, body.tracking_id);
  PutU32(out, body.enrollment_index);
  PutU32(out, body.user_index);
  PutVec4(out, body.position);
  for (const auto& joint : body.joints) {
    PutVec4(out, joint);
  }
  for (const auto& state : body.joint_states) {
    PutU32(out, static_cast<uint32_t>(state));
  }
  PutU32(out, body.quality_flags);
}

// |p| must point at kSkeletonBytes readable bytes.
void GetSkeleton(const uint8_t* p, Skeleton* out) {
  out->state = static_cast<SkeletonState>(GetU32(p));
  out->tracking_id = GetU32(p + 4);
  out->enrollment_index = GetU32(p + 8);
  out->user_index = GetU32(p + 12);
  out->position = GetVec4(p + 16);
  p += 32;
  for (auto& joint : out->joints) {
    joint = GetVec4(p);
    p += 16;
  }
  for (auto& state : out->joint_states) {
    state = static_cast<JointState>(GetU32(p));
    p += 4;
  }
  out->quality_flags = GetU32(p);
}

// Appends u32 size + zstd frame of |size| bytes at |data|.
bool PutCompressed(std::vector<uint8_t>* out, std::vector<uint8_t>* scratch,
                   const void* data, size_t size, std::string* out_error) {
  size_t bound = ZSTD_compressBound(size);
  scratch->resize(bound);
  size_t written =
      ZSTD_compress(scratch->data(), bound, data, size, kZstdLevel);
  if (ZSTD_isError(written)) {
    *out_error =
        std::string("zstd compression failed: ") + ZSTD_getErrorName(written);
    return false;
  }
  PutU32(out, static_cast<uint32_t>(written));
  out->insert(out->end(), scratch->begin(), scratch->begin() + written);
  return true;
}

}  // namespace

void NuirecFrame::Reset() {
  timestamp_us = 0;
  body_count = 0;
  for (auto& body : bodies) {
    body = Skeleton();
  }
  floor_clip_plane = Vec4();
  normal_to_gravity = Vec4();
  has_player_mask = false;
  has_depth = false;
  has_color = false;
}

void NuirecEncodeMask(const uint8_t* mask, size_t pixel_count,
                      std::vector<uint8_t>* out) {
  size_t count_offset = out->size();
  PutU32(out, 0);
  uint32_t run_count = 0;
  size_t i = 0;
  while (i < pixel_count) {
    uint8_t value = mask[i];
    size_t run = 1;
    while (i + run < pixel_count && mask[i + run] == value && run < 0xFFFF) {
      ++run;
    }
    out->push_back(value);
    PutU16(out, static_cast<uint16_t>(run));
    ++run_count;
    i += run;
  }
  uint8_t* p = out->data() + count_offset;
  for (int b = 0; b < 4; ++b) {
    p[b] = static_cast<uint8_t>((run_count >> (8 * b)) & 0xFF);
  }
}

size_t NuirecDecodeMask(const uint8_t* data, size_t size, size_t pixel_count,
                        uint8_t* out_mask, std::string* out_error) {
  if (size < 4) {
    *out_error = "mask: truncated run count";
    return 0;
  }
  uint32_t run_count = GetU32(data);
  size_t offset = 4;
  size_t pixel = 0;
  for (uint32_t r = 0; r < run_count; ++r) {
    if (offset + 3 > size) {
      *out_error = "mask: truncated run";
      return 0;
    }
    uint8_t value = data[offset];
    uint16_t length = GetU16(data + offset + 1);
    offset += 3;
    if (length == 0) {
      *out_error = "mask: zero-length run";
      return 0;
    }
    if (pixel + length > pixel_count) {
      *out_error = "mask: runs exceed the image size";
      return 0;
    }
    std::memset(out_mask + pixel, value, length);
    pixel += length;
  }
  if (pixel != pixel_count) {
    *out_error =
        fmt::format("mask: runs cover {} of {} pixels", pixel, pixel_count);
    return 0;
  }
  return offset;
}

// NuirecWriter ---------------------------------------------------------------

NuirecWriter::~NuirecWriter() { Close(); }

bool NuirecWriter::Open(const std::filesystem::path& path,
                        const NuirecHeader& header, std::string* out_error) {
  Close();
  if (path.empty()) {
    *out_error = "no path given";
    return false;
  }
  if (header.version != kNuirecVersion) {
    *out_error = fmt::format("unsupported version {}", header.version);
    return false;
  }
  auto parent = path.parent_path();
  if (!parent.empty() && !std::filesystem::exists(parent)) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }
  FILE* file = xe::filesystem::OpenFile(path, "wb");
  if (!file) {
    *out_error = fmt::format("cannot create '{}'", xe::path_to_utf8(path));
    return false;
  }
  buffer_.clear();
  buffer_.push_back('X');
  buffer_.push_back('N');
  buffer_.push_back('U');
  buffer_.push_back('I');
  PutU32(&buffer_, header.version);
  PutU32(&buffer_, header.flags);
  PutU32(&buffer_, header.fps);
  PutF32(&buffer_, header.camera_height_m);
  PutF32(&buffer_, header.hfov_degrees);
  if (fwrite(buffer_.data(), 1, buffer_.size(), file) != buffer_.size()) {
    fclose(file);
    *out_error = fmt::format("cannot write '{}'", xe::path_to_utf8(path));
    return false;
  }
  file_ = file;
  path_ = path;
  header_ = header;
  frames_written_ = 0;
  return true;
}

bool NuirecWriter::WriteFrame(const NuirecFrame& frame,
                              std::string* out_error) {
  if (!file_) {
    *out_error = "writer is not open";
    return false;
  }
  if (frame.body_count > kMaxSkeletons) {
    *out_error = fmt::format("body_count {} exceeds {}", frame.body_count,
                             kMaxSkeletons);
    return false;
  }
  buffer_.clear();
  PutU64(&buffer_, static_cast<uint64_t>(frame.timestamp_us));
  PutU32(&buffer_, frame.body_count);
  for (uint32_t i = 0; i < frame.body_count; ++i) {
    PutSkeleton(&buffer_, frame.bodies[i]);
  }
  PutVec4(&buffer_, frame.floor_clip_plane);
  PutVec4(&buffer_, frame.normal_to_gravity);

  if (header_.has_mask()) {
    if (frame.has_player_mask && frame.player_mask.size() >= kMaskPixels) {
      NuirecEncodeMask(frame.player_mask.data(), kMaskPixels, &buffer_);
    } else {
      zero_scratch_.assign(kMaskPixels, 0);
      NuirecEncodeMask(zero_scratch_.data(), kMaskPixels, &buffer_);
    }
  }
  if (header_.has_depth()) {
    const void* data = zero_scratch_.data();
    if (frame.has_depth && frame.depth_mm.size() >= kMaskPixels) {
      data = frame.depth_mm.data();
    } else {
      zero_scratch_.assign(kDepthBytes, 0);
      data = zero_scratch_.data();
    }
    if (!PutCompressed(&buffer_, &compress_scratch_, data, kDepthBytes,
                       out_error)) {
      return false;
    }
  }
  if (header_.has_color()) {
    const void* data = zero_scratch_.data();
    if (frame.has_color && frame.color_rgb.size() >= kColorBytes) {
      data = frame.color_rgb.data();
    } else {
      zero_scratch_.assign(kColorBytes, 0);
      data = zero_scratch_.data();
    }
    if (!PutCompressed(&buffer_, &compress_scratch_, data, kColorBytes,
                       out_error)) {
      return false;
    }
  }

  if (fwrite(buffer_.data(), 1, buffer_.size(), file_) != buffer_.size()) {
    *out_error = fmt::format("write error on '{}'", xe::path_to_utf8(path_));
    return false;
  }
  ++frames_written_;
  return true;
}

void NuirecWriter::Close() {
  if (file_) {
    fclose(file_);
    file_ = nullptr;
  }
  path_.clear();
}

// NuirecReader ---------------------------------------------------------------

NuirecReader::~NuirecReader() { Close(); }

bool NuirecReader::ReadExact(void* out, size_t size, bool* out_eof,
                             std::string* out_error) {
  *out_eof = false;
  size_t got = fread(out, 1, size, file_);
  if (got == size) {
    return true;
  }
  if (got == 0 && feof(file_)) {
    *out_eof = true;
    return false;
  }
  if (ferror(file_)) {
    *out_error = fmt::format("read error on '{}' after {} frames",
                             xe::path_to_utf8(path_), frames_read_);
  } else {
    *out_error = fmt::format("'{}' is truncated after {} frames",
                             xe::path_to_utf8(path_), frames_read_);
  }
  return false;
}

bool NuirecReader::Open(const std::filesystem::path& path,
                        std::string* out_error) {
  Close();
  if (path.empty()) {
    *out_error = "no path given";
    return false;
  }
  FILE* file = xe::filesystem::OpenFile(path, "rb");
  if (!file) {
    *out_error = fmt::format("cannot open '{}'", xe::path_to_utf8(path));
    return false;
  }
  uint8_t header[kNuirecHeaderSize];
  if (fread(header, 1, sizeof(header), file) != sizeof(header)) {
    fclose(file);
    *out_error = fmt::format("'{}' is too short to be a .nuirec file",
                             xe::path_to_utf8(path));
    return false;
  }
  if (std::memcmp(header, "XNUI", 4) != 0) {
    fclose(file);
    *out_error = fmt::format("'{}' is not a .nuirec file (bad magic)",
                             xe::path_to_utf8(path));
    return false;
  }
  NuirecHeader parsed;
  parsed.version = GetU32(header + 4);
  parsed.flags = GetU32(header + 8);
  parsed.fps = GetU32(header + 12);
  parsed.camera_height_m = GetF32(header + 16);
  parsed.hfov_degrees = GetF32(header + 20);
  if (parsed.version != kNuirecVersion) {
    fclose(file);
    *out_error = fmt::format("'{}': unsupported .nuirec version {}",
                             xe::path_to_utf8(path), parsed.version);
    return false;
  }
  if (parsed.flags & ~(kNuirecFlagMask | kNuirecFlagDepth | kNuirecFlagColor)) {
    fclose(file);
    *out_error = fmt::format("'{}': unknown header flags {:#x}",
                             xe::path_to_utf8(path), parsed.flags);
    return false;
  }
  if (parsed.fps == 0 || parsed.fps > 1000) {
    fclose(file);
    *out_error = fmt::format("'{}': bad frame rate {}", xe::path_to_utf8(path),
                             parsed.fps);
    return false;
  }
  file_ = file;
  path_ = path;
  header_ = parsed;
  frames_read_ = 0;
  return true;
}

bool NuirecReader::ReadFrame(NuirecFrame* out_frame, std::string* out_error) {
  out_error->clear();
  if (!file_) {
    *out_error = "reader is not open";
    return false;
  }
  bool eof = false;
  uint8_t fixed[12];
  if (!ReadExact(fixed, sizeof(fixed), &eof, out_error)) {
    return false;
  }
  out_frame->Reset();
  out_frame->timestamp_us = static_cast<int64_t>(GetU64(fixed));
  uint32_t body_count = GetU32(fixed + 8);
  if (body_count > kMaxSkeletons) {
    *out_error = fmt::format("'{}': frame {} has {} bodies (max {})",
                             xe::path_to_utf8(path_), frames_read_, body_count,
                             kMaxSkeletons);
    return false;
  }
  out_frame->body_count = body_count;
  buffer_.resize(body_count * kSkeletonBytes + 32);
  if (!ReadExact(buffer_.data(), buffer_.size(), &eof, out_error)) {
    if (eof) {
      *out_error = fmt::format("'{}' is truncated after {} frames",
                               xe::path_to_utf8(path_), frames_read_);
    }
    return false;
  }
  const uint8_t* p = buffer_.data();
  for (uint32_t i = 0; i < body_count; ++i) {
    GetSkeleton(p, &out_frame->bodies[i]);
    p += kSkeletonBytes;
  }
  out_frame->floor_clip_plane = GetVec4(p);
  out_frame->normal_to_gravity = GetVec4(p + 16);

  auto read_sized = [&](const char* what, std::vector<uint8_t>* into) {
    uint8_t size_bytes[4];
    if (!ReadExact(size_bytes, 4, &eof, out_error)) {
      if (eof) {
        *out_error = fmt::format("'{}': truncated {} in frame {}",
                                 xe::path_to_utf8(path_), what, frames_read_);
      }
      return false;
    }
    uint32_t size = GetU32(size_bytes);
    if (size == 0 || size > kMaxCompressedSize) {
      *out_error =
          fmt::format("'{}': bad {} size {} in frame {}",
                      xe::path_to_utf8(path_), what, size, frames_read_);
      return false;
    }
    into->resize(size);
    if (!ReadExact(into->data(), size, &eof, out_error)) {
      if (eof) {
        *out_error = fmt::format("'{}': truncated {} in frame {}",
                                 xe::path_to_utf8(path_), what, frames_read_);
      }
      return false;
    }
    return true;
  };

  if (header_.has_mask()) {
    // The run count tells how many bytes follow.
    uint8_t count_bytes[4];
    if (!ReadExact(count_bytes, 4, &eof, out_error)) {
      if (eof) {
        *out_error = fmt::format("'{}': truncated mask in frame {}",
                                 xe::path_to_utf8(path_), frames_read_);
      }
      return false;
    }
    uint32_t run_count = GetU32(count_bytes);
    if (run_count == 0 || run_count > kMaskPixels) {
      *out_error =
          fmt::format("'{}': bad mask run count {} in frame {}",
                      xe::path_to_utf8(path_), run_count, frames_read_);
      return false;
    }
    buffer_.resize(4 + static_cast<size_t>(run_count) * 3);
    std::memcpy(buffer_.data(), count_bytes, 4);
    if (!ReadExact(buffer_.data() + 4, buffer_.size() - 4, &eof, out_error)) {
      if (eof) {
        *out_error = fmt::format("'{}': truncated mask in frame {}",
                                 xe::path_to_utf8(path_), frames_read_);
      }
      return false;
    }
    out_frame->player_mask.resize(kMaskPixels);
    std::string mask_error;
    if (!NuirecDecodeMask(buffer_.data(), buffer_.size(), kMaskPixels,
                          out_frame->player_mask.data(), &mask_error)) {
      *out_error = fmt::format("'{}': frame {}: {}", xe::path_to_utf8(path_),
                               frames_read_, mask_error);
      return false;
    }
    out_frame->has_player_mask = true;
  }
  if (header_.has_depth()) {
    if (!read_sized("depth", &buffer_)) {
      return false;
    }
    out_frame->depth_mm.resize(kMaskPixels);
    size_t got = ZSTD_decompress(out_frame->depth_mm.data(), kDepthBytes,
                                 buffer_.data(), buffer_.size());
    if (ZSTD_isError(got) || got != kDepthBytes) {
      *out_error = fmt::format(
          "'{}': frame {}: depth decompression failed: {}",
          xe::path_to_utf8(path_), frames_read_,
          ZSTD_isError(got) ? ZSTD_getErrorName(got) : "wrong size");
      return false;
    }
    out_frame->has_depth = true;
  }
  if (header_.has_color()) {
    if (!read_sized("colour", &buffer_)) {
      return false;
    }
    out_frame->color_rgb.resize(kColorBytes);
    size_t got = ZSTD_decompress(out_frame->color_rgb.data(), kColorBytes,
                                 buffer_.data(), buffer_.size());
    if (ZSTD_isError(got) || got != kColorBytes) {
      *out_error = fmt::format(
          "'{}': frame {}: colour decompression failed: {}",
          xe::path_to_utf8(path_), frames_read_,
          ZSTD_isError(got) ? ZSTD_getErrorName(got) : "wrong size");
      return false;
    }
    out_frame->has_color = true;
  }
  ++frames_read_;
  return true;
}

bool NuirecReader::Rewind(std::string* out_error) {
  if (!file_) {
    *out_error = "reader is not open";
    return false;
  }
  if (!xe::filesystem::Seek(file_, static_cast<int64_t>(kNuirecHeaderSize),
                            SEEK_SET)) {
    *out_error = fmt::format("cannot seek in '{}'", xe::path_to_utf8(path_));
    return false;
  }
  frames_read_ = 0;
  return true;
}

void NuirecReader::Close() {
  if (file_) {
    fclose(file_);
    file_ = nullptr;
  }
  path_.clear();
}

}  // namespace nui
}  // namespace xe
