/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/filesystem.h"
#include "xenia/nui/nui_recorder.h"
#include "xenia/nui/nui_types.h"

namespace xe {
namespace nui {
namespace test {

namespace {

constexpr size_t kMaskPixels = static_cast<size_t>(kDepthWidth) * kDepthHeight;
constexpr size_t kColorBytes =
    static_cast<size_t>(kColorWidth) * kColorHeight * 3;

std::filesystem::path TempPath(const char* name) {
  std::error_code ec;
  auto dir = std::filesystem::temp_directory_path(ec);
  if (ec) {
    dir = std::filesystem::current_path();
  }
  return dir / name;
}

struct TempFile {
  explicit TempFile(const char* name) : path(TempPath(name)) {}
  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::filesystem::path path;
};

Vec4 MakeVec(float base) {
  return {base, base + 0.25f, base + 0.5f, base + 0.75f};
}

Skeleton MakeBody(uint32_t index, uint32_t frame) {
  Skeleton body;
  body.state =
      index == 0 ? SkeletonState::kTracked : SkeletonState::kPositionOnly;
  body.tracking_id = 100 + index;
  body.enrollment_index = index;
  body.user_index = index == 0 ? 0 : kInvalidUserIndex;
  body.position = MakeVec(static_cast<float>(frame) + 0.1f * index);
  for (uint32_t j = 0; j < kJointCount; ++j) {
    body.joints[j] = MakeVec(static_cast<float>(j) * 0.01f +
                             static_cast<float>(frame) + 10.0f * index);
    body.joint_states[j] =
        (j % 3 == 0) ? JointState::kInferred : JointState::kTracked;
  }
  body.quality_flags = kQualityClippedBottom | (frame & 1 ? 1u : 0u);
  return body;
}

NuirecFrame MakeFrame(uint32_t index, bool mask, bool depth, bool color) {
  NuirecFrame frame;
  frame.timestamp_us = 1000000 + static_cast<int64_t>(index) * 33333;
  frame.body_count = 1 + index;  // 1, 2, 3 bodies
  for (uint32_t b = 0; b < frame.body_count; ++b) {
    frame.bodies[b] = MakeBody(b, index);
  }
  frame.floor_clip_plane = {0.0f, 1.0f, 0.0f, 1.0f + 0.1f * index};
  frame.normal_to_gravity = {0.0f, 0.99f, 0.1f, 0.0f};
  if (mask) {
    frame.has_player_mask = true;
    frame.player_mask.assign(kMaskPixels, 0);
    // A rectangle per body plus a lone pixel so that runs of every length
    // class appear (including the > 65535 background run).
    for (uint32_t b = 0; b < frame.body_count; ++b) {
      for (uint32_t y = 40 + 20 * index; y < 200; ++y) {
        for (uint32_t x = 60 + 70 * b; x < 60 + 70 * b + 50; ++x) {
          frame.player_mask[y * kDepthWidth + x] = static_cast<uint8_t>(b + 1);
        }
      }
    }
    frame.player_mask[kMaskPixels - 1] = 6;
  }
  if (depth) {
    frame.has_depth = true;
    frame.depth_mm.resize(kMaskPixels);
    for (size_t i = 0; i < kMaskPixels; ++i) {
      frame.depth_mm[i] = static_cast<uint16_t>((i * 7 + index * 13) & 0x1FFF);
    }
  }
  if (color) {
    frame.has_color = true;
    frame.color_rgb.resize(kColorBytes);
    for (size_t i = 0; i < kColorBytes; ++i) {
      frame.color_rgb[i] = static_cast<uint8_t>((i / 3 + index) & 0xFF);
    }
  }
  return frame;
}

void RequireVecEqual(const Vec4& a, const Vec4& b) {
  REQUIRE(a.x == b.x);
  REQUIRE(a.y == b.y);
  REQUIRE(a.z == b.z);
  REQUIRE(a.w == b.w);
}

void RequireBodyEqual(const Skeleton& a, const Skeleton& b) {
  REQUIRE(a.state == b.state);
  REQUIRE(a.tracking_id == b.tracking_id);
  REQUIRE(a.enrollment_index == b.enrollment_index);
  REQUIRE(a.user_index == b.user_index);
  RequireVecEqual(a.position, b.position);
  for (uint32_t j = 0; j < kJointCount; ++j) {
    RequireVecEqual(a.joints[j], b.joints[j]);
    REQUIRE(a.joint_states[j] == b.joint_states[j]);
  }
  REQUIRE(a.quality_flags == b.quality_flags);
}

void RequireFrameEqual(const NuirecFrame& written, const NuirecFrame& read) {
  REQUIRE(read.timestamp_us == written.timestamp_us);
  REQUIRE(read.body_count == written.body_count);
  for (uint32_t b = 0; b < written.body_count; ++b) {
    RequireBodyEqual(written.bodies[b], read.bodies[b]);
  }
  RequireVecEqual(written.floor_clip_plane, read.floor_clip_plane);
  RequireVecEqual(written.normal_to_gravity, read.normal_to_gravity);
  REQUIRE(read.has_player_mask == written.has_player_mask);
  if (written.has_player_mask) {
    REQUIRE(read.player_mask.size() == kMaskPixels);
    REQUIRE(read.player_mask == written.player_mask);
  }
  REQUIRE(read.has_depth == written.has_depth);
  if (written.has_depth) {
    REQUIRE(read.depth_mm.size() == kMaskPixels);
    REQUIRE(read.depth_mm == written.depth_mm);
  }
  REQUIRE(read.has_color == written.has_color);
  if (written.has_color) {
    REQUIRE(read.color_rgb.size() == kColorBytes);
    REQUIRE(read.color_rgb == written.color_rgb);
  }
}

}  // namespace

TEST_CASE("Mask RLE round-trips and splits long runs", "[nui][nuirec]") {
  std::vector<uint8_t> mask(kMaskPixels, 0);
  for (size_t i = 1000; i < 2000; ++i) {
    mask[i] = 1;
  }
  mask[kMaskPixels - 1] = 2;
  std::vector<uint8_t> encoded;
  NuirecEncodeMask(mask.data(), kMaskPixels, &encoded);
  // 1000 zeros, 1000 ones, 74799 zeros (65535 + 9264), one 2 = 5 runs.
  REQUIRE(encoded.size() == 4 + 5 * 3);
  std::vector<uint8_t> decoded(kMaskPixels, 0xAA);
  std::string error;
  REQUIRE(NuirecDecodeMask(encoded.data(), encoded.size(), kMaskPixels,
                           decoded.data(), &error) == encoded.size());
  REQUIRE(error.empty());
  REQUIRE(decoded == mask);

  // Runs that do not cover the image are rejected.
  encoded.resize(encoded.size() - 3);
  encoded[0] = 4;
  REQUIRE(NuirecDecodeMask(encoded.data(), encoded.size(), kMaskPixels,
                           decoded.data(), &error) == 0);
  REQUIRE_FALSE(error.empty());
}

TEST_CASE("nuirec writes and reads back three frames", "[nui][nuirec]") {
  TempFile file("xenia_nuirec_test_full.nuirec");
  NuirecHeader header;
  header.flags = kNuirecFlagMask | kNuirecFlagDepth | kNuirecFlagColor;
  header.fps = 30;
  header.camera_height_m = 1.25f;
  header.hfov_degrees = 70.0f;

  std::vector<NuirecFrame> frames;
  for (uint32_t i = 0; i < 3; ++i) {
    frames.push_back(MakeFrame(i, true, true, true));
  }

  std::string error;
  {
    NuirecWriter writer;
    REQUIRE(writer.Open(file.path, header, &error));
    REQUIRE(error.empty());
    for (const auto& frame : frames) {
      REQUIRE(writer.WriteFrame(frame, &error));
    }
    REQUIRE(writer.frames_written() == 3);
    writer.Close();
    REQUIRE_FALSE(writer.is_open());
  }

  NuirecReader reader;
  REQUIRE(reader.Open(file.path, &error));
  REQUIRE(reader.header().version == kNuirecVersion);
  REQUIRE(reader.header().flags == header.flags);
  REQUIRE(reader.header().fps == 30);
  REQUIRE(reader.header().camera_height_m == 1.25f);
  REQUIRE(reader.header().hfov_degrees == 70.0f);

  for (uint32_t pass = 0; pass < 2; ++pass) {
    for (const auto& written : frames) {
      NuirecFrame read;
      REQUIRE(reader.ReadFrame(&read, &error));
      REQUIRE(error.empty());
      RequireFrameEqual(written, read);
    }
    NuirecFrame extra;
    REQUIRE_FALSE(reader.ReadFrame(&extra, &error));
    REQUIRE(error.empty());  // clean end of file
    REQUIRE(reader.frames_read() == 3);
    REQUIRE(reader.Rewind(&error));
  }
  reader.Close();
}

TEST_CASE("nuirec skeleton-only file and missing sections", "[nui][nuirec]") {
  TempFile file("xenia_nuirec_test_skel.nuirec");
  NuirecHeader header;
  header.flags = kNuirecFlagDepth;  // depth declared, mask/colour not
  std::string error;
  NuirecFrame with_depth = MakeFrame(0, true, true, true);
  NuirecFrame without_depth = MakeFrame(1, false, false, false);
  {
    NuirecWriter writer;
    REQUIRE(writer.Open(file.path, header, &error));
    REQUIRE(writer.WriteFrame(with_depth, &error));
    REQUIRE(writer.WriteFrame(without_depth, &error));
  }
  NuirecReader reader;
  REQUIRE(reader.Open(file.path, &error));
  NuirecFrame read;
  REQUIRE(reader.ReadFrame(&read, &error));
  // Mask and colour are not in the file even though the frame had them.
  REQUIRE_FALSE(read.has_player_mask);
  REQUIRE_FALSE(read.has_color);
  REQUIRE(read.has_depth);
  REQUIRE(read.depth_mm == with_depth.depth_mm);
  REQUIRE(read.body_count == 1);
  RequireBodyEqual(with_depth.bodies[0], read.bodies[0]);

  REQUIRE(reader.ReadFrame(&read, &error));
  REQUIRE(read.body_count == 2);
  REQUIRE(read.has_depth);
  // A frame without depth is stored as zeros.
  for (uint16_t value : read.depth_mm) {
    REQUIRE(value == 0);
  }
  REQUIRE_FALSE(reader.ReadFrame(&read, &error));
  REQUIRE(error.empty());
}

TEST_CASE("nuirec rejects bad and truncated files", "[nui][nuirec]") {
  std::string error;
  {
    NuirecReader reader;
    REQUIRE_FALSE(reader.Open(TempPath("xenia_nuirec_missing.nuirec"), &error));
    REQUIRE_FALSE(error.empty());
  }
  TempFile bad("xenia_nuirec_test_bad.nuirec");
  {
    FILE* f = xe::filesystem::OpenFile(bad.path, "wb");
    REQUIRE(f != nullptr);
    const char junk[] = "NOPE\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";
    fwrite(junk, 1, kNuirecHeaderSize, f);
    fclose(f);
    NuirecReader reader;
    REQUIRE_FALSE(reader.Open(bad.path, &error));
    REQUIRE(error.find("magic") != std::string::npos);
  }
  TempFile truncated("xenia_nuirec_test_trunc.nuirec");
  {
    NuirecHeader header;
    header.flags = kNuirecFlagMask;
    NuirecWriter writer;
    REQUIRE(writer.Open(truncated.path, header, &error));
    REQUIRE(writer.WriteFrame(MakeFrame(0, true, false, false), &error));
    REQUIRE(writer.WriteFrame(MakeFrame(1, true, false, false), &error));
    writer.Close();
    std::error_code ec;
    auto size = std::filesystem::file_size(truncated.path, ec);
    REQUIRE_FALSE(ec);
    std::filesystem::resize_file(truncated.path, size - 5, ec);
    REQUIRE_FALSE(ec);
    NuirecReader reader;
    REQUIRE(reader.Open(truncated.path, &error));
    NuirecFrame frame;
    REQUIRE(reader.ReadFrame(&frame, &error));  // first frame intact
    REQUIRE_FALSE(reader.ReadFrame(&frame, &error));
    REQUIRE(error.find("truncated") != std::string::npos);
  }
}

}  // namespace test
}  // namespace nui
}  // namespace xe
