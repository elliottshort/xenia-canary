/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_NUI_NUI_TYPES_H_
#define XENIA_NUI_NUI_TYPES_H_

#include <array>
#include <cstdint>
#include <vector>

// Host-side representation of what a Kinect (NUI) sensor delivers to a title.
// Everything here is in host byte order and host units; the guest-facing
// layer (src/xenia/kernel/nui) converts to the big-endian structures of the
// NUI runtime version linked into the running title.
//
// Conventions (Kinect for Xbox 360 / Kinect SDK v1 "skeleton space"):
//   - metres, right-handed, origin at the sensor;
//   - +X points to the sensor's left (the user's right when facing it);
//   - +Y points up;
//   - +Z points away from the sensor, towards the user.
// Depth images are "mirror" views: a user's right hand appears on the right
// side of the image.

namespace xe {
namespace nui {

// NUI_SKELETON_POSITION_INDEX, in Kinect SDK v1 order.
enum class Joint : uint8_t {
  kHipCenter = 0,
  kSpine,
  kShoulderCenter,
  kHead,
  kShoulderLeft,
  kElbowLeft,
  kWristLeft,
  kHandLeft,
  kShoulderRight,
  kElbowRight,
  kWristRight,
  kHandRight,
  kHipLeft,
  kKneeLeft,
  kAnkleLeft,
  kFootLeft,
  kHipRight,
  kKneeRight,
  kAnkleRight,
  kFootRight,
  kCount,
};
constexpr uint32_t kJointCount = static_cast<uint32_t>(Joint::kCount);

// NUI_SKELETON_POSITION_TRACKING_STATE
enum class JointState : uint32_t {
  kNotTracked = 0,
  kInferred = 1,
  kTracked = 2,
};

// NUI_SKELETON_TRACKING_STATE
enum class SkeletonState : uint32_t {
  kNotTracked = 0,
  kPositionOnly = 1,
  kTracked = 2,
};

// NUI_SKELETON_QUALITY_*
enum QualityFlags : uint32_t {
  kQualityClippedRight = 0x00000001,
  kQualityClippedLeft = 0x00000002,
  kQualityClippedTop = 0x00000004,
  kQualityClippedBottom = 0x00000008,
};

// NUI_INITIALIZE_FLAG_*
enum InitializeFlags : uint32_t {
  kInitDepthAndPlayerIndex = 0x00000001,
  kInitColor = 0x00000002,
  kInitSkeleton = 0x00000008,
  kInitDepth = 0x00000020,
  kInitHighQualityColor = 0x00000040,
  kInitAudio = 0x10000000,
  kInitKnownMask = kInitDepthAndPlayerIndex | kInitColor | kInitSkeleton |
                   kInitDepth | kInitHighQualityColor | kInitAudio,
};

// NUI_SKELETON_TRACKING_FLAG_* (Kinect SDK v1.5+ values; the flag set the
// running NUI library version accepts is decided by the guest-facing layer).
enum SkeletonTrackingFlags : uint32_t {
  kTrackingSuppressNoFrameData = 0x00000001,
  kTrackingTitleSetsTrackedSkeletons = 0x00000002,
  kTrackingEnableSeatedSupport = 0x00000004,
  kTrackingEnableInNearRange = 0x00000008,
};

// NUI_IMAGE_TYPE
enum class ImageType : uint32_t {
  kDepthAndPlayerIndex = 0,
  kColor = 1,
  kColorYuv = 2,
  kColorRawYuv = 3,
  kDepth = 4,
  kColorInfrared = 5,
  kColorRawBayer = 6,
};

// NUI_IMAGE_RESOLUTION
enum class ImageResolution : uint32_t {
  k80x60 = 0,
  k320x240 = 1,
  k640x480 = 2,
  k1280x960 = 3,
};

// NUI_IMAGE_STREAM_FLAG_* / NUI_IMAGE_FRAME_FLAG_*
enum ImageStreamFlags : uint32_t {
  kStreamSuppressNoFrameData = 0x00010000,
  kStreamEnableNearMode = 0x00020000,
  kStreamDistinctOverflowDepthValues = 0x00040000,
};

inline bool ImageTypeIsDepth(ImageType type) {
  return type == ImageType::kDepthAndPlayerIndex || type == ImageType::kDepth;
}

inline uint32_t ImageResolutionWidth(ImageResolution resolution) {
  switch (resolution) {
    case ImageResolution::k80x60:
      return 80;
    case ImageResolution::k320x240:
      return 320;
    case ImageResolution::k640x480:
      return 640;
    case ImageResolution::k1280x960:
      return 1280;
  }
  return 0;
}

inline uint32_t ImageResolutionHeight(ImageResolution resolution) {
  switch (resolution) {
    case ImageResolution::k80x60:
      return 60;
    case ImageResolution::k320x240:
      return 240;
    case ImageResolution::k640x480:
      return 480;
    case ImageResolution::k1280x960:
      return 960;
  }
  return 0;
}

struct Vec4 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
};

constexpr uint32_t kMaxSkeletons = 6;
constexpr uint32_t kMaxTrackedSkeletons = 2;
constexpr uint32_t kInvalidTrackingId = 0;
constexpr uint32_t kInvalidUserIndex = 0xFFFFFFFFu;
constexpr uint32_t kFrameRateHz = 30;
constexpr uint32_t kFramePeriodMicroseconds = 33333;

// Depth camera model (nominal Kinect v1 intrinsics; NuiSensor.h).
constexpr float kDepthNominalFocalLengthPx = 285.63f;  // at 320x240
constexpr float kColorNominalFocalLengthPx = 531.15f;  // at 640x480
constexpr float kDepthHorizontalFovDegrees = 58.5f;
constexpr float kDepthVerticalFovDegrees = 45.6f;
constexpr float kColorHorizontalFovDegrees = 62.0f;
constexpr uint16_t kDepthMinMillimetres = 800;
constexpr uint16_t kDepthMaxMillimetres = 4000;
constexpr uint16_t kDepthNearMinMillimetres = 400;
constexpr uint16_t kDepthNearMaxMillimetres = 3000;
constexpr uint16_t kDepthUnknown = 0;
constexpr uint16_t kDepthTooFar = 0x0FFF;
constexpr uint32_t kPlayerIndexBits = 3;
constexpr uint16_t kPlayerIndexMask = (1u << kPlayerIndexBits) - 1;

// Working resolution of the host pipeline: depth/player-index images are
// produced at the sensor's native tracking resolution and derived from there.
constexpr uint32_t kDepthWidth = 320;
constexpr uint32_t kDepthHeight = 240;
constexpr uint32_t kColorWidth = 640;
constexpr uint32_t kColorHeight = 480;

// One skeleton, mirroring NUI_SKELETON_DATA.
struct Skeleton {
  SkeletonState state = SkeletonState::kNotTracked;
  uint32_t tracking_id = kInvalidTrackingId;
  uint32_t enrollment_index = kInvalidUserIndex;
  uint32_t user_index = kInvalidUserIndex;
  Vec4 position;
  std::array<Vec4, kJointCount> joints{};
  std::array<JointState, kJointCount> joint_states{};
  uint32_t quality_flags = 0;
};

// One published skeleton frame, mirroring NUI_SKELETON_FRAME.
struct SkeletonFrame {
  // Microseconds since NuiInitialize (host monotonic, paused with the
  // emulator). The guest-facing layer converts to the unit the title expects.
  int64_t timestamp_us = 0;
  uint32_t frame_number = 0;
  uint32_t flags = 0;
  // Ax + By + Cz + D = 0, normalized so that D is the sensor height above the
  // floor in metres. Zero when the floor is unknown.
  Vec4 floor_clip_plane;
  // Unit vector pointing "up" (against gravity) in sensor space.
  Vec4 normal_to_gravity;
  std::array<Skeleton, kMaxSkeletons> skeletons{};
};

// NUI_TRANSFORM_SMOOTH_PARAMETERS
struct SmoothParameters {
  float smoothing = 0.5f;
  float correction = 0.5f;
  float prediction = 0.5f;
  float jitter_radius = 0.05f;
  float max_deviation_radius = 0.04f;
};

// One published image frame. Depth frames carry unpacked millimetres plus a
// separate player-index plane; colour frames carry 0xAARRGGBB pixels in host
// order. The guest-facing layer packs/byte-swaps into the texture layout the
// title expects.
struct ImageFrame {
  ImageType type = ImageType::kDepthAndPlayerIndex;
  ImageResolution resolution = ImageResolution::k320x240;
  uint32_t width = 0;
  uint32_t height = 0;
  int64_t timestamp_us = 0;
  uint32_t frame_number = 0;
  uint32_t flags = 0;
  std::vector<uint16_t> depth_mm;        // width * height (depth types only)
  std::vector<uint8_t> player_index;     // width * height (depth types only)
  std::vector<uint32_t> color_argb;      // width * height (colour types only)
};

// What a NuiSource produces: a fully synthesized sensor observation in
// skeleton space. Skeleton tracking ids here are source-local "person keys":
// stable while the same person is tracked, never reused by that source. The
// NuiSystem maps them to the tracking ids and skeleton slots a title sees.
struct SourceFrame {
  uint64_t sequence = 0;
  // Host steady-clock time of the underlying camera capture, in microseconds.
  uint64_t capture_time_us = 0;
  // Bodies detected by the source, tracked ones first.
  uint32_t body_count = 0;
  std::array<Skeleton, kMaxSkeletons> bodies{};
  Vec4 floor_clip_plane;
  Vec4 normal_to_gravity;
  // Player mask at kDepthWidth x kDepthHeight: 0 = background, otherwise the
  // 1-based index into |bodies| of the person covering that pixel.
  bool has_player_mask = false;
  std::vector<uint8_t> player_mask;
  // Depth at kDepthWidth x kDepthHeight in millimetres, 0 = unknown.
  bool has_depth = false;
  std::vector<uint16_t> depth_mm;
  // Colour at kColorWidth x kColorHeight, 0xAARRGGBB host order.
  bool has_color = false;
  std::vector<uint32_t> color_argb;

  void Reset() {
    sequence = 0;
    capture_time_us = 0;
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
};

}  // namespace nui
}  // namespace xe

#endif  // XENIA_NUI_NUI_TYPES_H_
