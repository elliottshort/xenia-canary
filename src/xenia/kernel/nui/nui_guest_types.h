/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_NUI_NUI_GUEST_TYPES_H_
#define XENIA_KERNEL_NUI_NUI_GUEST_TYPES_H_

#include <cstdint>

#include "xenia/base/assert.h"
#include "xenia/base/byte_order.h"

// Big-endian guest layouts of the public NUI API structures (Kinect SDK v1
// / Xbox 360 XDK nuiapi). Offsets verified against the Project Milo build
// (XDK 11427) and the Kinect for Windows SDK headers.

namespace xe {
namespace kernel {
namespace nui {

struct X_NUI_VECTOR4 {
  xe::be<float> x;
  xe::be<float> y;
  xe::be<float> z;
  xe::be<float> w;
};
static_assert_size(X_NUI_VECTOR4, 0x10);

constexpr uint32_t kGuestJointCount = 20;
constexpr uint32_t kGuestSkeletonCount = 6;

struct X_NUI_SKELETON_DATA {
  xe::be<uint32_t> tracking_state;     // 0x00 NUI_SKELETON_TRACKING_STATE
  xe::be<uint32_t> tracking_id;        // 0x04
  xe::be<uint32_t> enrollment_index;   // 0x08
  xe::be<uint32_t> user_index;         // 0x0C
  X_NUI_VECTOR4 position;              // 0x10
  X_NUI_VECTOR4 joints[kGuestJointCount];           // 0x20
  xe::be<uint32_t> joint_states[kGuestJointCount];  // 0x160
  xe::be<uint32_t> quality_flags;      // 0x1B0
  uint8_t padding[0xC];                // 0x1B4 (16-byte alignment)
};
static_assert_size(X_NUI_SKELETON_DATA, 0x1C0);

struct X_NUI_SKELETON_FRAME {
  xe::be<int64_t> timestamp;           // 0x00
  xe::be<uint32_t> frame_number;       // 0x08
  xe::be<uint32_t> flags;              // 0x0C
  X_NUI_VECTOR4 floor_clip_plane;      // 0x10
  X_NUI_VECTOR4 normal_to_gravity;     // 0x20
  X_NUI_SKELETON_DATA skeletons[kGuestSkeletonCount];  // 0x30
};
static_assert_size(X_NUI_SKELETON_FRAME, 0xAB0);

struct X_NUI_TRANSFORM_SMOOTH_PARAMETERS {
  xe::be<float> smoothing;
  xe::be<float> correction;
  xe::be<float> prediction;
  xe::be<float> jitter_radius;
  xe::be<float> max_deviation_radius;
};
static_assert_size(X_NUI_TRANSFORM_SMOOTH_PARAMETERS, 0x14);

struct X_NUI_IMAGE_VIEW_AREA {
  xe::be<uint32_t> digital_zoom;
  xe::be<int32_t> center_x;
  xe::be<int32_t> center_y;
};
static_assert_size(X_NUI_IMAGE_VIEW_AREA, 0xC);

// The frame descriptor handed to titles by NuiImageStreamGetNextFrame. The
// public part (through view_area) matches the SDK; the tail mirrors the
// private fields of the XDK 11427 runtime's descriptor so that title code
// compiled against that SDK finds the pixel pointer where it expects it.
struct X_NUI_IMAGE_FRAME {
  xe::be<int64_t> timestamp;           // 0x00
  xe::be<uint32_t> frame_number;       // 0x08
  xe::be<uint32_t> image_type;         // 0x0C NUI_IMAGE_TYPE
  xe::be<uint32_t> resolution;         // 0x10 NUI_IMAGE_RESOLUTION
  xe::be<uint32_t> frame_texture_ptr;  // 0x14 D3DTexture*
  xe::be<uint32_t> frame_flags;        // 0x18
  X_NUI_IMAGE_VIEW_AREA view_area;     // 0x1C
  xe::be<uint32_t> reserved[11];       // 0x28
  xe::be<uint32_t> pixels_ptr;         // 0x54 (XDK 11427 private field)
  xe::be<uint32_t> reserved2[2];       // 0x58
};
static_assert_size(X_NUI_IMAGE_FRAME, 0x60);

// Xbox 360 D3DBaseTexture: the D3DResource header followed by the GPU
// texture fetch constant. Titles hand it to the GPU or to D3DTexture_LockRect
// (which reads the fetch constant) and D3DResource_BlockUntilNotBusy (which
// reads |fence|).
struct X_D3D_TEXTURE {
  xe::be<uint32_t> common;           // 0x00
  xe::be<uint32_t> reference_count;  // 0x04
  xe::be<uint32_t> fence;            // 0x08
  xe::be<uint32_t> read_fence;       // 0x0C
  xe::be<uint32_t> identifier;       // 0x10
  xe::be<uint32_t> base_flush;       // 0x14
  xe::be<uint32_t> mip_flush;        // 0x18
  xe::be<uint32_t> fetch[6];         // 0x1C xe_gpu_texture_fetch_t
  xe::be<uint32_t> padding[3];       // 0x34 -> 0x40
};
static_assert_size(X_D3D_TEXTURE, 0x40);

}  // namespace nui
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_NUI_NUI_GUEST_TYPES_H_
