/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/nui/nui_hle_handlers.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/logging.h"
#include "xenia/base/math.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/emulator.h"
#include "xenia/gpu/xenos.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/nui/nui_guest_types.h"
#include "xenia/kernel/nui/nui_hle.h"
#include "xenia/kernel/util/object_table.h"
#include "xenia/kernel/xevent.h"
#include "xenia/kernel/xobject.h"
#include "xenia/memory.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/nui_system.h"
#include "xenia/nui/nui_types.h"

namespace xe {
namespace kernel {
namespace nui {

namespace {

using cpu::ppc::PPCContext;

constexpr uint32_t kMaxWaitMilliseconds = 8000;
constexpr uint32_t kTextureRowAlignment = 256;
constexpr uint32_t kPixelBufferAlignment = 4096;

// One image stream the title opened through the HLE'd API.
struct HleStream {
  uint32_t handle = 0;
  object_ref<XEvent> handle_object;
  uint32_t system_stream_id = 0;
  xe::nui::ImageType type = xe::nui::ImageType::kDepthAndPlayerIndex;
  xe::nui::ImageResolution resolution = xe::nui::ImageResolution::k320x240;
  uint32_t frame_limit = 2;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t bytes_per_pixel = 2;
  uint32_t pitch = 0;
  uint32_t slot_bytes = 0;
  uint32_t descriptors_ptr = 0;
  uint32_t textures_ptr = 0;
  uint32_t pixels_ptr = 0;
  uint32_t pixels_phys = 0;
  std::vector<uint8_t> held;
  uint32_t last_frame_number = 0;
  object_ref<XEvent> next_frame_event;
  xe::nui::ImageFrame scratch;
};

// Guest allocations are pooled for the lifetime of the title: titles keep
// frame/texture pointers around and the GPU may still read them.
struct GuestBuffers {
  uint32_t frame_limit = 0;
  uint32_t descriptors_ptr = 0;
  uint32_t textures_ptr = 0;
  uint32_t pixels_ptr = 0;
  uint32_t pixels_phys = 0;
  uint32_t slot_bytes = 0;
};

struct HleState {
  std::mutex mutex;
  KernelState* kernel_state = nullptr;
  uint32_t listener_id = 0;
  object_ref<XEvent> skeleton_event;
  object_ref<XEvent> frame_end_event;
  uint32_t last_skeleton_frame_number = 0;
  std::vector<std::unique_ptr<HleStream>> streams;
  std::map<uint64_t, GuestBuffers> pools;
  // A title-specific hook layer owns a stateful part of the runtime and
  // reads the device model directly; see SetNuiHleExternalOwner.
  bool external_owner = false;
  bool logged_first_skeleton = false;
  bool logged_first_image = false;
};

HleState& State() {
  static HleState state;
  return state;
}

xe::nui::NuiSystem* NuiSystemOf(PPCContext* ctx) {
  return ctx->kernel_state->emulator()->nui_system();
}

void Return(PPCContext* ctx, uint32_t hresult) {
  ctx->r[3] = static_cast<uint64_t>(
      static_cast<int64_t>(static_cast<int32_t>(hresult)));
}

uint32_t Arg(PPCContext* ctx, int index) {
  return static_cast<uint32_t>(ctx->r[3 + index]);
}

uint32_t ClampWait(uint32_t timeout_ms) {
  if (timeout_ms == 0xFFFFFFFFu) {
    return timeout_ms;
  }
  return std::min(timeout_ms, kMaxWaitMilliseconds);
}

void Trace(const char* name, PPCContext* ctx, int arg_count) {
  if (!cvars::nui_trace) {
    return;
  }
  std::string args;
  for (int i = 0; i < arg_count; ++i) {
    args += fmt::format("{}{:08X}", i ? ", " : "", Arg(ctx, i));
  }
  XELOGI("NuiHLE: {}({})", name, args);
}

// Signals the guest events titles wait on, on the pacer thread.
void OnFramePublished() {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.skeleton_event) {
    state.skeleton_event->Set(0, false);
  }
  for (auto& stream : state.streams) {
    if (stream->next_frame_event) {
      stream->next_frame_event->Set(0, false);
    }
  }
  if (state.frame_end_event) {
    state.frame_end_event->Set(0, false);
  }
}

void EnsureListener(KernelState* kernel_state, xe::nui::NuiSystem* nui) {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.kernel_state = kernel_state;
  if (!state.listener_id) {
    state.listener_id = nui->AddFrameListener(OnFramePublished);
  }
}

void WriteVector4(X_NUI_VECTOR4* dst, const xe::nui::Vec4& src) {
  dst->x = src.x;
  dst->y = src.y;
  dst->z = src.z;
  dst->w = src.w;
}

xe::nui::Vec4 ReadVector4(const X_NUI_VECTOR4* src) {
  return {src->x, src->y, src->z, src->w};
}

void WriteSkeletonFrame(X_NUI_SKELETON_FRAME* dst,
                        const xe::nui::SkeletonFrame& src) {
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  std::memset(dst, 0, sizeof(*dst));
  dst->timestamp =
      quirks.timestamp_100ns ? src.timestamp_us * 10 : src.timestamp_us / 1000;
  dst->frame_number = src.frame_number;
  dst->flags = src.flags;
  xe::nui::Vec4 floor = src.floor_clip_plane;
  if (quirks.floor_plane_millimetres) {
    floor.w *= 1000.0f;
  }
  WriteVector4(&dst->floor_clip_plane, floor);
  WriteVector4(&dst->normal_to_gravity, src.normal_to_gravity);
  for (uint32_t i = 0; i < kGuestSkeletonCount; ++i) {
    const auto& s = src.skeletons[i];
    auto& d = dst->skeletons[i];
    d.tracking_state = static_cast<uint32_t>(s.state);
    d.tracking_id = s.tracking_id;
    d.enrollment_index = s.enrollment_index;
    d.user_index = s.user_index;
    WriteVector4(&d.position, s.position);
    for (uint32_t j = 0; j < kGuestJointCount; ++j) {
      WriteVector4(&d.joints[j], s.joints[j]);
      d.joint_states[j] = static_cast<uint32_t>(s.joint_states[j]);
    }
    d.quality_flags = s.quality_flags;
  }
}

void ReadSkeletonFrame(const X_NUI_SKELETON_FRAME* src,
                       xe::nui::SkeletonFrame* dst) {
  dst->frame_number = src->frame_number;
  dst->flags = src->flags;
  dst->floor_clip_plane = ReadVector4(&src->floor_clip_plane);
  dst->normal_to_gravity = ReadVector4(&src->normal_to_gravity);
  for (uint32_t i = 0; i < kGuestSkeletonCount; ++i) {
    const auto& s = src->skeletons[i];
    auto& d = dst->skeletons[i];
    d.state = static_cast<xe::nui::SkeletonState>(uint32_t(s.tracking_state));
    d.tracking_id = s.tracking_id;
    d.enrollment_index = s.enrollment_index;
    d.user_index = s.user_index;
    d.position = ReadVector4(&s.position);
    for (uint32_t j = 0; j < kGuestJointCount; ++j) {
      d.joints[j] = ReadVector4(&s.joints[j]);
      d.joint_states[j] =
          static_cast<xe::nui::JointState>(uint32_t(s.joint_states[j]));
    }
    d.quality_flags = s.quality_flags;
  }
}

HleStream* FindStream(HleState& state, uint32_t handle) {
  for (auto& stream : state.streams) {
    if (stream->handle == handle) {
      return stream.get();
    }
  }
  return nullptr;
}

// Fills the Xbox 360 D3D texture header for one frame slot: a linear
// (untiled) 2D texture whose fetch constant points at the slot's pixels.
void WriteTextureHeader(X_D3D_TEXTURE* tex, const HleStream& stream,
                        uint32_t pixels_phys) {
  using namespace xe::gpu::xenos;
  const bool is_depth = xe::nui::ImageTypeIsDepth(stream.type);
  xe_gpu_texture_fetch_t fetch = {};
  fetch.type = FetchConstantType::kTexture;
  fetch.sign_x = TextureSign::kUnsigned;
  fetch.sign_y = TextureSign::kUnsigned;
  fetch.sign_z = TextureSign::kUnsigned;
  fetch.sign_w = TextureSign::kUnsigned;
  fetch.clamp_x = ClampMode::kClampToEdge;
  fetch.clamp_y = ClampMode::kClampToEdge;
  fetch.clamp_z = ClampMode::kClampToEdge;
  fetch.pitch = stream.pitch >> 5;
  fetch.tiled = 0;
  fetch.format = is_depth ? TextureFormat::k_16 : TextureFormat::k_8_8_8_8;
  fetch.endianness = is_depth ? Endian::k8in16 : Endian::k8in32;
  fetch.base_address = pixels_phys >> 12;
  fetch.size_2d.width = stream.width - 1;
  fetch.size_2d.height = stream.height - 1;
  fetch.size_2d.stack_depth = 0;
  // Component swizzle: D3DFMT_A8R8G8B8 is ZYXW, D3DFMT_L16 is XXX1.
  fetch.swizzle = is_depth ? (0 | (0 << 3) | (0 << 6) | (5 << 9))
                           : (2 | (1 << 3) | (0 << 6) | (3 << 9));
  fetch.mag_filter = TextureFilter::kLinear;
  fetch.min_filter = TextureFilter::kLinear;
  fetch.mip_filter = TextureFilter::kBaseMap;
  fetch.aniso_filter = AnisoFilter::kDisabled;
  fetch.dimension = DataDimension::k2DOrStacked;

  std::memset(tex, 0, sizeof(*tex));
  // D3DResource header: type = texture, one reference, no pending GPU
  // fences (D3DResource_BlockUntilNotBusy returns immediately), flush words
  // in the "nothing to flush" state.
  tex->common = (3u << 16) | 1u;
  tex->reference_count = 1;
  tex->fence = 0;
  tex->read_fence = 0;
  tex->identifier = 0;
  tex->base_flush = 0xFFFF0000u;
  tex->mip_flush = 0xFFFF0000u;
  tex->fetch[0] = fetch.dword_0;
  tex->fetch[1] = fetch.dword_1;
  tex->fetch[2] = fetch.dword_2;
  tex->fetch[3] = fetch.dword_3;
  tex->fetch[4] = fetch.dword_4;
  tex->fetch[5] = fetch.dword_5;
}

bool AllocateStreamBuffers(KernelState* kernel_state, HleState& state,
                           HleStream* stream) {
  Memory* memory = kernel_state->memory();
  stream->width = xe::nui::ImageResolutionWidth(stream->resolution);
  stream->height = xe::nui::ImageResolutionHeight(stream->resolution);
  stream->bytes_per_pixel = xe::nui::ImageTypeIsDepth(stream->type) ? 2 : 4;
  stream->pitch =
      xe::align(stream->width * stream->bytes_per_pixel, kTextureRowAlignment);
  stream->slot_bytes =
      xe::align(stream->pitch * stream->height, kPixelBufferAlignment);

  const uint64_t key = (uint64_t(stream->type) << 40) |
                       (uint64_t(stream->resolution) << 32) |
                       stream->frame_limit;
  auto it = state.pools.find(key);
  if (it == state.pools.end()) {
    GuestBuffers buffers;
    buffers.frame_limit = stream->frame_limit;
    buffers.slot_bytes = stream->slot_bytes;
    buffers.descriptors_ptr = memory->SystemHeapAlloc(
        stream->frame_limit * uint32_t(sizeof(X_NUI_IMAGE_FRAME)), 0x40);
    buffers.textures_ptr = memory->SystemHeapAlloc(
        stream->frame_limit * uint32_t(sizeof(X_D3D_TEXTURE)), 0x40);
    buffers.pixels_ptr =
        memory->SystemHeapAlloc(stream->frame_limit * stream->slot_bytes,
                                kPixelBufferAlignment, kSystemHeapPhysical);
    if (!buffers.descriptors_ptr || !buffers.textures_ptr ||
        !buffers.pixels_ptr) {
      XELOGE("NuiHLE: failed to allocate guest memory for an image stream");
      return false;
    }
    buffers.pixels_phys = memory->GetPhysicalAddress(buffers.pixels_ptr);
    memory->Zero(buffers.descriptors_ptr,
                 stream->frame_limit * uint32_t(sizeof(X_NUI_IMAGE_FRAME)));
    memory->Zero(buffers.textures_ptr,
                 stream->frame_limit * uint32_t(sizeof(X_D3D_TEXTURE)));
    memory->Zero(buffers.pixels_ptr, stream->frame_limit * stream->slot_bytes);
    it = state.pools.emplace(key, buffers).first;
    XELOGI(
        "NuiHLE: image stream buffers: {} frames of {}x{}x{} at {:08X} "
        "(phys {:08X}), descriptors {:08X}, textures {:08X}",
        stream->frame_limit, stream->width, stream->height,
        stream->bytes_per_pixel, buffers.pixels_ptr, buffers.pixels_phys,
        buffers.descriptors_ptr, buffers.textures_ptr);
  }
  const GuestBuffers& buffers = it->second;
  stream->descriptors_ptr = buffers.descriptors_ptr;
  stream->textures_ptr = buffers.textures_ptr;
  stream->pixels_ptr = buffers.pixels_ptr;
  stream->pixels_phys = buffers.pixels_phys;
  stream->held.assign(stream->frame_limit, 0);
  for (uint32_t i = 0; i < stream->frame_limit; ++i) {
    auto* tex = memory->TranslateVirtual<X_D3D_TEXTURE*>(
        stream->textures_ptr + i * uint32_t(sizeof(X_D3D_TEXTURE)));
    WriteTextureHeader(tex, *stream,
                       stream->pixels_phys + i * stream->slot_bytes);
  }
  return true;
}

// Copies a host image frame into a guest slot in the big-endian texture
// layout the title expects, and fills the slot's descriptor.
void FillSlot(KernelState* kernel_state, HleStream* stream, uint32_t slot,
              const xe::nui::ImageFrame& frame) {
  Memory* memory = kernel_state->memory();
  uint8_t* pixels =
      memory->TranslateVirtual(stream->pixels_ptr + slot * stream->slot_bytes);
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  if (xe::nui::ImageTypeIsDepth(stream->type)) {
    const bool distinct_overflow =
        (stream->scratch.flags & xe::nui::kStreamDistinctOverflowDepthValues) !=
        0;
    for (uint32_t y = 0; y < stream->height; ++y) {
      uint8_t* row = pixels + y * stream->pitch;
      for (uint32_t x = 0; x < stream->width; ++x) {
        const size_t i = size_t(y) * stream->width + x;
        uint16_t depth = frame.depth_mm.empty() ? 0 : frame.depth_mm[i];
        uint16_t player =
            frame.player_index.empty() ? 0 : frame.player_index[i];
        uint16_t packed;
        if (depth == 0) {
          packed = 0;
        } else if (depth > xe::nui::kDepthMaxMillimetres) {
          packed = distinct_overflow ? uint16_t(xe::nui::kDepthTooFar
                                                << xe::nui::kPlayerIndexBits)
                                     : 0;
        } else {
          packed = uint16_t((depth << xe::nui::kPlayerIndexBits) |
                            (player & xe::nui::kPlayerIndexMask));
        }
        xe::store_and_swap<uint16_t>(row + x * 2, packed);
      }
    }
  } else {
    for (uint32_t y = 0; y < stream->height; ++y) {
      uint8_t* row = pixels + y * stream->pitch;
      for (uint32_t x = 0; x < stream->width; ++x) {
        const size_t i = size_t(y) * stream->width + x;
        uint32_t argb = frame.color_argb.empty() ? 0 : frame.color_argb[i];
        xe::store_and_swap<uint32_t>(row + x * 4, argb);
      }
    }
  }

  auto* desc = memory->TranslateVirtual<X_NUI_IMAGE_FRAME*>(
      stream->descriptors_ptr + slot * uint32_t(sizeof(X_NUI_IMAGE_FRAME)));
  desc->timestamp = quirks.timestamp_100ns ? frame.timestamp_us * 10
                                           : frame.timestamp_us / 1000;
  desc->frame_number = frame.frame_number;
  desc->image_type = static_cast<uint32_t>(stream->type);
  desc->resolution = static_cast<uint32_t>(stream->resolution);
  desc->frame_texture_ptr =
      stream->textures_ptr + slot * uint32_t(sizeof(X_D3D_TEXTURE));
  desc->frame_flags = frame.flags;
  desc->view_area.digital_zoom = 1;
  desc->view_area.center_x = 0;
  desc->view_area.center_y = 0;
  desc->pixels_ptr = stream->pixels_ptr + slot * stream->slot_bytes;
}

// ---------------------------------------------------------------------------
// Handlers. Arguments follow the PPC ABI (r3..r10); results go to r3.
// ---------------------------------------------------------------------------

// HRESULT NuiInitialize(DWORD dwFlags)
void NuiInitialize(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiInitialize", ctx, 1);
  auto* nui = NuiSystemOf(ctx);
  if (!nui) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  const uint32_t flags = Arg(ctx, 0);
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  if (!flags || (flags & ~quirks.init_flags_mask)) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      XELOGW(
          "NuiHLE: NuiInitialize({:08X}) rejected: unknown flag bits "
          "(accepted mask {:08X})",
          flags, quirks.init_flags_mask);
    }
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  // Translate to the host flag set: depth/colour bits are shared, skeletal
  // tracking may be requested through version-specific bits.
  uint32_t host_flags = flags & xe::nui::kInitKnownMask &
                        ~static_cast<uint32_t>(xe::nui::kInitSkeleton);
  if (flags & quirks.init_skeleton_flags) {
    host_flags |= xe::nui::kInitSkeleton;
  }
  if (host_flags & (xe::nui::kInitDepth | xe::nui::kInitSkeleton)) {
    // Depth and player index are one stream on the sensor.
    host_flags |= xe::nui::kInitDepthAndPlayerIndex | xe::nui::kInitDepth;
  }
  uint32_t hr = nui->Initialize(host_flags);
  if (hr == xe::nui::kNuiErrorAlreadyInitialized) {
    static bool logged_twice = false;
    if (!logged_twice) {
      logged_twice = true;
      XELOGW("NuiHLE: NuiInitialize called while already initialized");
    }
  }
  if (hr == xe::nui::kNuiOk) {
    EnsureListener(kernel_state, nui);
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.last_skeleton_frame_number = nui->current_frame_number();
  }
  Return(ctx, hr);
}

// void NuiShutdown()
void NuiShutdown(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiShutdown", ctx, 0);
  auto* nui = NuiSystemOf(ctx);
  bool external_owner = false;
  {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    external_owner = state.external_owner;
    for (auto& stream : state.streams) {
      if (nui) {
        nui->CloseImageStream(stream->system_stream_id);
      }
    }
    state.streams.clear();
    state.skeleton_event.reset();
    state.frame_end_event.reset();
  }
  if (nui && !external_owner) {
    nui->Uninitialize();
  } else if (nui) {
    // Somebody else owns a stateful part of the runtime and reads the device
    // model without going through us: stopping it here would reopen the
    // camera and reload the inference models on that owner's next frame,
    // seconds of work on a guest thread. Only our own resources are dropped.
    XELOGI(
        "NuiHLE: NuiShutdown: the device model stays up for the title hooks "
        "that own part of the runtime");
  }
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiSkeletonTrackingEnable(HANDLE hNextFrameEvent, DWORD dwFlags)
void NuiSkeletonTrackingEnable(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiSkeletonTrackingEnable", ctx, 2);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t event_handle = Arg(ctx, 0);
  const uint32_t flags = Arg(ctx, 1);
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  constexpr uint32_t kKnownFlags = xe::nui::kTrackingSuppressNoFrameData |
                                   xe::nui::kTrackingTitleSetsTrackedSkeletons |
                                   xe::nui::kTrackingEnableSeatedSupport |
                                   xe::nui::kTrackingEnableInNearRange;
  if (!nui) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  if (flags & ~kKnownFlags) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  object_ref<XEvent> event;
  if (event_handle) {
    event = kernel_state->object_table()->LookupObject<XEvent>(event_handle);
    if (!event) {
      XELOGW("NuiSkeletonTrackingEnable: bad event handle {:08X}",
             event_handle);
      Return(ctx, xe::nui::kNuiErrorInvalidArg);
      return;
    }
  }
  uint32_t hr = nui->EnableSkeletonTracking(flags);
  if (hr == xe::nui::kNuiErrorFeatureNotInitialized) {
    hr = quirks.hr_feature_not_initialized;
  }
  if (hr == xe::nui::kNuiOk) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.skeleton_event = event;
    state.last_skeleton_frame_number = nui->current_frame_number();
  }
  Return(ctx, hr);
}

// HRESULT NuiSkeletonTrackingDisable()
void NuiSkeletonTrackingDisable(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiSkeletonTrackingDisable", ctx, 0);
  auto* nui = NuiSystemOf(ctx);
  if (nui) {
    nui->DisableSkeletonTracking();
  }
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.skeleton_event.reset();
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiSkeletonGetNextFrame(DWORD dwMillisecondsToWait,
//                                 NUI_SKELETON_FRAME* pSkeletonFrame)
void NuiSkeletonGetNextFrame(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiSkeletonGetNextFrame", ctx, 2);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t timeout_ms = ClampWait(Arg(ctx, 0));
  const uint32_t frame_ptr = Arg(ctx, 1);
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  if (!frame_ptr) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  if (!nui || !nui->initialized()) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  if (!nui->skeleton_tracking_enabled()) {
    Return(ctx, quirks.hr_stream_not_enabled);
    return;
  }
  uint32_t last_number;
  {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    last_number = state.last_skeleton_frame_number;
  }
  xe::nui::SkeletonFrame frame;
  if (!nui->GetNextSkeletonFrame(last_number, timeout_ms, &frame)) {
    Return(ctx, quirks.hr_wait_timeout);
    return;
  }
  {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.last_skeleton_frame_number = frame.frame_number;
    if (!state.logged_first_skeleton) {
      state.logged_first_skeleton = true;
      XELOGI("NuiHLE: first skeleton frame delivered (frame {})",
             frame.frame_number);
    }
  }
  if (nui->skeleton_tracking_flags() & xe::nui::kTrackingSuppressNoFrameData) {
    bool anybody = false;
    for (const auto& skeleton : frame.skeletons) {
      if (skeleton.state != xe::nui::SkeletonState::kNotTracked) {
        anybody = true;
        break;
      }
    }
    if (!anybody) {
      Return(ctx, xe::nui::kNuiErrorFrameNoData);
      return;
    }
  }
  WriteSkeletonFrame(ctx->TranslateVirtual<X_NUI_SKELETON_FRAME*>(frame_ptr),
                     frame);
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiSkeletonSetTrackedSkeletons(DWORD TrackingIDs[2])
void NuiSkeletonSetTrackedSkeletons(PPCContext* ctx,
                                    KernelState* kernel_state) {
  Trace("NuiSkeletonSetTrackedSkeletons", ctx, 1);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t ids_ptr = Arg(ctx, 0);
  if (!ids_ptr) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  const uint8_t* p = ctx->TranslateVirtual<const uint8_t*>(ids_ptr);
  const uint32_t first = xe::load_and_swap<uint32_t>(p);
  const uint32_t second = xe::load_and_swap<uint32_t>(p + 4);
  if (nui) {
    nui->SetTrackedSkeletons(first, second);
  }
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiTransformSmooth(NUI_SKELETON_FRAME* pSkeletonFrame,
//                            const NUI_TRANSFORM_SMOOTH_PARAMETERS* pParams)
void NuiTransformSmooth(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiTransformSmooth", ctx, 2);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t frame_ptr = Arg(ctx, 0);
  const uint32_t params_ptr = Arg(ctx, 1);
  if (!frame_ptr || !nui) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  auto* guest = ctx->TranslateVirtual<X_NUI_SKELETON_FRAME*>(frame_ptr);
  xe::nui::SkeletonFrame frame;
  ReadSkeletonFrame(guest, &frame);
  xe::nui::SmoothParameters params;
  const xe::nui::SmoothParameters* params_ptr_host = nullptr;
  if (params_ptr) {
    auto* p =
        ctx->TranslateVirtual<X_NUI_TRANSFORM_SMOOTH_PARAMETERS*>(params_ptr);
    params.smoothing = p->smoothing;
    params.correction = p->correction;
    params.prediction = p->prediction;
    params.jitter_radius = p->jitter_radius;
    params.max_deviation_radius = p->max_deviation_radius;
    params_ptr_host = &params;
  }
  nui->TransformSmooth(&frame, params_ptr_host);
  for (uint32_t i = 0; i < kGuestSkeletonCount; ++i) {
    if (frame.skeletons[i].state != xe::nui::SkeletonState::kTracked) {
      continue;
    }
    for (uint32_t j = 0; j < kGuestJointCount; ++j) {
      WriteVector4(&guest->skeletons[i].joints[j],
                   frame.skeletons[i].joints[j]);
    }
  }
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiImageStreamOpen(NUI_IMAGE_TYPE eImageType,
//                            NUI_IMAGE_RESOLUTION eResolution,
//                            DWORD dwImageFrameFlags, DWORD dwFrameLimit,
//                            HANDLE hNextFrameEvent, HANDLE* phStreamHandle)
void NuiImageStreamOpen(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiImageStreamOpen", ctx, 6);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t type = Arg(ctx, 0);
  const uint32_t resolution = Arg(ctx, 1);
  const uint32_t flags = Arg(ctx, 2);
  const uint32_t frame_limit = Arg(ctx, 3);
  const uint32_t event_handle = Arg(ctx, 4);
  const uint32_t out_handle_ptr = Arg(ctx, 5);
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  if (!out_handle_ptr) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  xe::store_and_swap<uint32_t>(ctx->TranslateVirtual<uint8_t*>(out_handle_ptr),
                               0);
  if (!nui) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  object_ref<XEvent> event;
  if (event_handle) {
    event = kernel_state->object_table()->LookupObject<XEvent>(event_handle);
    if (!event) {
      XELOGW("NuiImageStreamOpen: bad event handle {:08X}", event_handle);
      Return(ctx, xe::nui::kNuiErrorInvalidArg);
      return;
    }
  }
  uint32_t system_stream_id = 0;
  uint32_t hr = nui->OpenImageStream(
      static_cast<xe::nui::ImageType>(type),
      static_cast<xe::nui::ImageResolution>(resolution), flags,
      frame_limit ? frame_limit : 2, &system_stream_id);
  if (hr == xe::nui::kNuiErrorImageStreamInUse) {
    hr = quirks.hr_image_stream_in_use;
  } else if (hr == xe::nui::kNuiErrorFrameLimitExceeded) {
    hr = quirks.hr_frame_limit_exceeded;
  } else if (hr == xe::nui::kNuiErrorFeatureNotInitialized) {
    hr = quirks.hr_feature_not_initialized;
  }
  if (hr != xe::nui::kNuiOk) {
    Return(ctx, hr);
    return;
  }

  auto stream = std::make_unique<HleStream>();
  stream->system_stream_id = system_stream_id;
  stream->type = static_cast<xe::nui::ImageType>(type);
  stream->resolution = static_cast<xe::nui::ImageResolution>(resolution);
  stream->frame_limit = frame_limit ? frame_limit : 2;
  stream->next_frame_event = event;
  stream->last_frame_number = nui->current_frame_number();
  // The stream handle is a real kernel object so NtClose and
  // ObReferenceObjectByHandle keep working on it.
  stream->handle_object = object_ref<XEvent>(new XEvent(kernel_state));
  stream->handle_object->Initialize(true, false);
  stream->handle = stream->handle_object->handle();

  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.kernel_state = kernel_state;
  if (!AllocateStreamBuffers(kernel_state, state, stream.get())) {
    nui->CloseImageStream(system_stream_id);
    Return(ctx, xe::nui::kNuiErrorDeviceNotReady);
    return;
  }
  xe::store_and_swap<uint32_t>(ctx->TranslateVirtual<uint8_t*>(out_handle_ptr),
                               stream->handle);
  XELOGI("NuiHLE: image stream opened: handle {:08X} type {} resolution {}",
         stream->handle, type, resolution);
  state.streams.push_back(std::move(stream));
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiImageStreamGetNextFrame(HANDLE hStream, DWORD
// dwMillisecondsToWait,
//                                    const NUI_IMAGE_FRAME** ppcImageFrame)
void NuiImageStreamGetNextFrame(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiImageStreamGetNextFrame", ctx, 3);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t handle = Arg(ctx, 0);
  const uint32_t timeout_ms = ClampWait(Arg(ctx, 1));
  const uint32_t out_frame_ptr = Arg(ctx, 2);
  const NuiHleQuirks& quirks = GetActiveNuiHleQuirks();
  if (!handle || !out_frame_ptr) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  if (!nui || !nui->initialized()) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  auto& state = State();
  HleStream* stream;
  uint32_t system_stream_id;
  uint32_t last_number;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    stream = FindStream(state, handle);
    if (!stream) {
      Return(ctx, xe::nui::kNuiErrorInvalidArg);
      return;
    }
    uint32_t held = 0;
    for (uint8_t h : stream->held) {
      held += h;
    }
    if (held >= stream->frame_limit) {
      Return(ctx, quirks.hr_frame_limit_exceeded);
      return;
    }
    system_stream_id = stream->system_stream_id;
    last_number = stream->last_frame_number;
  }
  // Wait outside the lock; the stream itself only goes away on shutdown.
  xe::nui::ImageFrame scratch;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    scratch = std::move(stream->scratch);
  }
  const bool got = nui->GetNextImageFrame(system_stream_id, last_number,
                                          timeout_ms, &scratch);
  std::lock_guard<std::mutex> lock(state.mutex);
  stream = FindStream(state, handle);
  if (!stream) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  stream->scratch = std::move(scratch);
  if (!got) {
    Return(ctx, quirks.hr_wait_timeout);
    return;
  }
  stream->last_frame_number = stream->scratch.frame_number;
  uint32_t slot = stream->frame_limit;
  for (uint32_t i = 0; i < stream->frame_limit; ++i) {
    if (!stream->held[i]) {
      slot = i;
      break;
    }
  }
  if (slot == stream->frame_limit) {
    Return(ctx, quirks.hr_frame_limit_exceeded);
    return;
  }
  FillSlot(kernel_state, stream, slot, stream->scratch);
  stream->held[slot] = 1;
  const uint32_t descriptor =
      stream->descriptors_ptr + slot * uint32_t(sizeof(X_NUI_IMAGE_FRAME));
  xe::store_and_swap<uint32_t>(ctx->TranslateVirtual<uint8_t*>(out_frame_ptr),
                               descriptor);
  if (!state.logged_first_image) {
    state.logged_first_image = true;
    XELOGI("NuiHLE: first image frame delivered (stream {:08X}, frame {})",
           handle, stream->scratch.frame_number);
  }
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiImageStreamReleaseFrame(HANDLE hStream,
//                                    const NUI_IMAGE_FRAME* pImageFrame)
void NuiImageStreamReleaseFrame(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiImageStreamReleaseFrame", ctx, 2);
  const uint32_t handle = Arg(ctx, 0);
  const uint32_t frame_ptr = Arg(ctx, 1);
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  HleStream* stream = FindStream(state, handle);
  if (!stream || !frame_ptr || frame_ptr < stream->descriptors_ptr) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  const uint32_t offset = frame_ptr - stream->descriptors_ptr;
  const uint32_t slot = offset / uint32_t(sizeof(X_NUI_IMAGE_FRAME));
  if (offset % sizeof(X_NUI_IMAGE_FRAME) != 0 || slot >= stream->frame_limit ||
      !stream->held[slot]) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  stream->held[slot] = 0;
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiImageStreamSetImageFrameFlags(HANDLE hStream, DWORD dwFlags)
void NuiImageStreamSetImageFrameFlags(PPCContext* ctx,
                                      KernelState* kernel_state) {
  Trace("NuiImageStreamSetImageFrameFlags", ctx, 2);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t handle = Arg(ctx, 0);
  const uint32_t flags = Arg(ctx, 1);
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  HleStream* stream = FindStream(state, handle);
  if (!stream || !nui ||
      !nui->SetImageStreamFlags(stream->system_stream_id, flags)) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiImageStreamGetImageFrameFlags(HANDLE hStream, DWORD* pdwFlags)
void NuiImageStreamGetImageFrameFlags(PPCContext* ctx,
                                      KernelState* kernel_state) {
  Trace("NuiImageStreamGetImageFrameFlags", ctx, 2);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t handle = Arg(ctx, 0);
  const uint32_t out_ptr = Arg(ctx, 1);
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  HleStream* stream = FindStream(state, handle);
  uint32_t flags = 0;
  if (!stream || !out_ptr || !nui ||
      !nui->GetImageStreamFlags(stream->system_stream_id, &flags)) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  xe::store_and_swap<uint32_t>(ctx->TranslateVirtual<uint8_t*>(out_ptr), flags);
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiSetFrameEndEvent(HANDLE hEvent, DWORD dwFrameEventFlag)
//
// The event this arms is signalled by our pacer (OnFramePublished), not by
// an image stream, so its state is global and it belongs in HookSet::kNone
// rather than in the image set: a title whose image set is ceded but whose
// skeleton set is ours still needs it. No signature table has an entry for
// it yet; see docs/nui/architecture.md.
void NuiSetFrameEndEvent(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiSetFrameEndEvent", ctx, 2);
  const uint32_t event_handle = Arg(ctx, 0);
  object_ref<XEvent> event;
  if (event_handle) {
    event = kernel_state->object_table()->LookupObject<XEvent>(event_handle);
    if (!event) {
      Return(ctx, xe::nui::kNuiErrorInvalidArg);
      return;
    }
  }
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.frame_end_event = event;
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiCameraElevationSetAngle(LONG lAngleDegrees)
void NuiCameraElevationSetAngle(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiCameraElevationSetAngle", ctx, 1);
  auto* nui = NuiSystemOf(ctx);
  const int32_t degrees = static_cast<int32_t>(Arg(ctx, 0));
  if (!nui) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  Return(ctx, nui->SetElevationAngle(degrees));
}

// HRESULT NuiCameraElevationGetAngle(LONG* plAngleDegrees)
void NuiCameraElevationGetAngle(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiCameraElevationGetAngle", ctx, 1);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t out_ptr = Arg(ctx, 0);
  if (!out_ptr) {
    Return(ctx, xe::nui::kNuiErrorPointer);
    return;
  }
  if (!nui) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  xe::store_and_swap<int32_t>(ctx->TranslateVirtual<uint8_t*>(out_ptr),
                              nui->elevation_angle());
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiCameraGetNormalToGravity(Vector4* pNormalToGravity)
void NuiCameraGetNormalToGravity(PPCContext* ctx, KernelState* kernel_state) {
  Trace("NuiCameraGetNormalToGravity", ctx, 1);
  auto* nui = NuiSystemOf(ctx);
  const uint32_t out_ptr = Arg(ctx, 0);
  if (!out_ptr) {
    Return(ctx, xe::nui::kNuiErrorPointer);
    return;
  }
  if (!nui) {
    Return(ctx, xe::nui::kNuiErrorDeviceNotConnected);
    return;
  }
  WriteVector4(ctx->TranslateVirtual<X_NUI_VECTOR4*>(out_ptr),
               nui->normal_to_gravity());
  Return(ctx, xe::nui::kNuiOk);
}

// HRESULT NuiImageGetColorPixelCoordinatesFromDepthPixel(
//     NUI_IMAGE_RESOLUTION eColorResolution,
//     const NUI_IMAGE_VIEW_AREA* pcViewArea, LONG lDepthX, LONG lDepthY,
//     USHORT usDepthValue, LONG* plColorX, LONG* plColorY)
void NuiImageGetColorPixelCoordinatesFromDepthPixel(PPCContext* ctx,
                                                    KernelState* kernel_state) {
  Trace("NuiImageGetColorPixelCoordinatesFromDepthPixel", ctx, 7);
  const uint32_t color_resolution = Arg(ctx, 0);
  const int32_t depth_x = static_cast<int32_t>(Arg(ctx, 2));
  const int32_t depth_y = static_cast<int32_t>(Arg(ctx, 3));
  const uint32_t out_x = Arg(ctx, 5);
  const uint32_t out_y = Arg(ctx, 6);
  if (!out_x || !out_y) {
    Return(ctx, xe::nui::kNuiErrorPointer);
    return;
  }
  const uint32_t color_width = xe::nui::ImageResolutionWidth(
      static_cast<xe::nui::ImageResolution>(color_resolution));
  const uint32_t color_height = xe::nui::ImageResolutionHeight(
      static_cast<xe::nui::ImageResolution>(color_resolution));
  if (!color_width) {
    Return(ctx, xe::nui::kNuiErrorInvalidArg);
    return;
  }
  // The depth coordinates are in whatever resolution the title opened its
  // depth stream at (80x60 and 160x120 are accepted), not in the working
  // resolution of the pipeline.
  uint32_t depth_width = xe::nui::kDepthWidth;
  uint32_t depth_height = xe::nui::kDepthHeight;
  {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    for (const auto& stream : state.streams) {
      if (xe::nui::ImageTypeIsDepth(stream->type) && stream->width &&
          stream->height) {
        depth_width = stream->width;
        depth_height = stream->height;
        break;
      }
    }
  }
  // The synthesized colour and depth images share one point of view, so the
  // mapping is a pure resolution scale.
  const int32_t color_x = depth_x * static_cast<int32_t>(color_width) /
                          static_cast<int32_t>(depth_width);
  const int32_t color_y = depth_y * static_cast<int32_t>(color_height) /
                          static_cast<int32_t>(depth_height);
  xe::store_and_swap<int32_t>(ctx->TranslateVirtual<uint8_t*>(out_x), color_x);
  xe::store_and_swap<int32_t>(ctx->TranslateVirtual<uint8_t*>(out_y), color_y);
  Return(ctx, xe::nui::kNuiOk);
}

struct HandlerEntry {
  const char* name;
  cpu::GuestFunction::ExternHandler handler;
};

const HandlerEntry kHandlers[] = {
    {"NuiInitialize", NuiInitialize},
    {"NuiShutdown", NuiShutdown},
    {"NuiSkeletonTrackingEnable", NuiSkeletonTrackingEnable},
    {"NuiSkeletonTrackingDisable", NuiSkeletonTrackingDisable},
    {"NuiSkeletonGetNextFrame", NuiSkeletonGetNextFrame},
    {"NuiSkeletonSetTrackedSkeletons", NuiSkeletonSetTrackedSkeletons},
    {"NuiTransformSmooth", NuiTransformSmooth},
    {"NuiImageStreamOpen", NuiImageStreamOpen},
    {"NuiImageStreamGetNextFrame", NuiImageStreamGetNextFrame},
    {"NuiImageStreamReleaseFrame", NuiImageStreamReleaseFrame},
    {"NuiImageStreamSetImageFrameFlags", NuiImageStreamSetImageFrameFlags},
    {"NuiImageStreamGetImageFrameFlags", NuiImageStreamGetImageFrameFlags},
    {"NuiSetFrameEndEvent", NuiSetFrameEndEvent},
    {"NuiCameraElevationSetAngle", NuiCameraElevationSetAngle},
    {"NuiCameraElevationGetAngle", NuiCameraElevationGetAngle},
    {"NuiCameraGetNormalToGravity", NuiCameraGetNormalToGravity},
    {"NuiImageGetColorPixelCoordinatesFromDepthPixel",
     NuiImageGetColorPixelCoordinatesFromDepthPixel},
};

// Tripwires: known API surface we do not emulate yet.
constexpr const char* kTripwireNames[] = {
    "NuiSpeechEnable",           "NuiSpeechLoadGrammarFromMemory",
    "NuiSpeechStartRecognition", "NuiSpeechStopRecognition",
    "NuiSpeechUnloadGrammar",    "NuiSpeechSetEventInterest",
    "NuiSpeechGetEvents",        "NuiSpeechDestroyEvent",
    "NuiHandsInitialize",        "NuiHandsReset",
    "NuiHandsShutdown",          "NuiIdentityEnroll",
    "NuiIdentityIdentify",       "NuiSkeletonCalculateBoneOrientations",
    "NuiAudioMicArrayStart",     "NuiAudioMicArrayStop",
};
constexpr size_t kTripwireCount =
    sizeof(kTripwireNames) / sizeof(kTripwireNames[0]);

template <size_t I>
void Tripwire(PPCContext* ctx, KernelState* kernel_state) {
  static bool logged = false;
  if (!logged) {
    logged = true;
    XELOGW(
        "NuiHLE: title called unsupported Kinect function {} "
        "(r3={:08X} r4={:08X} r5={:08X}); returning E_NOTIMPL",
        kTripwireNames[I], Arg(ctx, 0), Arg(ctx, 1), Arg(ctx, 2));
  }
  Return(ctx, xe::nui::kNuiErrorNotImplemented);
}

template <size_t... I>
constexpr std::array<cpu::GuestFunction::ExternHandler, sizeof...(I)>
MakeTripwires(std::index_sequence<I...>) {
  return {{&Tripwire<I>...}};
}

const auto kTripwires =
    MakeTripwires(std::make_index_sequence<kTripwireCount>{});

}  // namespace

cpu::GuestFunction::ExternHandler LookupNuiHleHandler(std::string_view name) {
  for (const auto& entry : kHandlers) {
    if (name == entry.name) {
      return entry.handler;
    }
  }
  return nullptr;
}

cpu::GuestFunction::ExternHandler LookupNuiTripwireHandler(
    std::string_view name) {
  for (size_t i = 0; i < kTripwireCount; ++i) {
    if (name == kTripwireNames[i]) {
      return kTripwires[i];
    }
  }
  return nullptr;
}

void SetNuiHleExternalOwner(bool external_owner) {
  auto& state = State();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.external_owner = external_owner;
}

void ResetNuiHleState(KernelState* kernel_state, bool free_guest_memory) {
  auto& state = State();
  auto* nui = kernel_state->emulator()->nui_system();
  uint32_t listener_id = 0;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    listener_id = state.listener_id;
    state.listener_id = 0;
  }
  // Outside state.mutex: the pacer holds its listener list while it calls
  // OnFramePublished, which takes state.mutex.
  if (nui && listener_id) {
    nui->RemoveFrameListener(listener_id);
  }
  std::lock_guard<std::mutex> lock(state.mutex);
  state.kernel_state = kernel_state;
  state.skeleton_event.reset();
  state.frame_end_event.reset();
  state.last_skeleton_frame_number = 0;
  state.streams.clear();
  state.external_owner = false;
  if (free_guest_memory) {
    // The title that owns these is being terminated and its memory is still
    // ours to release; a colour stream is up to 4 * 640 * 480 * 4 bytes of
    // guest physical memory, which would otherwise stay allocated for the
    // rest of the process.
    Memory* memory = kernel_state->memory();
    for (auto& entry : state.pools) {
      if (entry.second.descriptors_ptr) {
        memory->SystemHeapFree(entry.second.descriptors_ptr);
      }
      if (entry.second.textures_ptr) {
        memory->SystemHeapFree(entry.second.textures_ptr);
      }
      if (entry.second.pixels_ptr) {
        memory->SystemHeapFree(entry.second.pixels_ptr);
      }
    }
  }
  state.pools.clear();
  state.logged_first_skeleton = false;
  state.logged_first_image = false;
}

}  // namespace nui
}  // namespace kernel
}  // namespace xe
