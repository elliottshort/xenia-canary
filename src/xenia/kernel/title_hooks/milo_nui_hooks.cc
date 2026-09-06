/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/title_hooks/milo_nui_hooks.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "xenia/base/byte_order.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/base/threading.h"
#include "xenia/cpu/ppc/ppc_context.h"
#include "xenia/cpu/xex_module.h"
#include "xenia/emulator.h"
#include "xenia/hid/input.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/title_hooks/virtual_kinect.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/util/object_table.h"
#include "xenia/kernel/xobject.h"
#include "xenia/nui/nui_system.h"
#include "xenia/nui/nui_types.h"

DEFINE_bool(virtual_kinect, true,
            "Emulate a Kinect sensor for supported titles by replacing the "
            "NUI runtime entry points inside the title with host code.",
            "Kernel");
DEFINE_bool(milo_disable_dof, false,
            "Project Milo: turn off the title's depth-of-field post-process "
            "(diagnostic).",
            "Kernel");
DEFINE_bool(milo_debug_dumps, false,
            "Project Milo: log the title's Kinect/cursor state and the host "
            "gamepad state every 5 seconds (diagnostic).",
            "Kernel");

namespace xe {
namespace kernel {
namespace hooks {
namespace milo {

namespace {

using cpu::ppc::PPCContext;

// HRESULTs returned by the NUI runtime library (nuiapi) for frame requests.
constexpr uint32_t kHResultPending = 0x8000000A;  // E_PENDING: wait timed out
constexpr uint32_t kHResultNoData = 0x83010001;   // E_NUI_FRAME_NO_DATA
constexpr uint32_t kHResultInvalidArg = 0x80070057;

// Paces a title thread that polls a frame function: sleeps for the requested
// timeout (bounded, so that INFINITE never blocks the emulator) and reports
// that no frame arrived. The real runtime does the same wait on a kernel event
// before returning E_PENDING.
uint32_t WaitNoFrame(uint32_t timeout_ms) {
  uint32_t sleep_ms = timeout_ms == 0xFFFFFFFF ? 33u : std::min(timeout_ms, 1000u);
  if (sleep_ms) {
    xe::threading::Sleep(std::chrono::milliseconds(sleep_ms));
  }
  return kHResultPending;
}

uint32_t Load32(PPCContext* ctx, uint32_t address);

// Diagnostic: periodically log the title's LHNatal state so the tracking mode
// it runs in (full skeleton vs. seated "NUI hands") and its health are visible.
void DumpNatalState(PPCContext* ctx) {
  if (!cvars::milo_debug_dumps) {
    return;
  }
  static std::chrono::steady_clock::time_point last;
  auto now = std::chrono::steady_clock::now();
  if (now - last < std::chrono::seconds(5)) {
    return;
  }
  last = now;
  // LHNatal statics (miloReleaseLIB build).
  const uint8_t* natal = ctx->TranslateVirtual<const uint8_t*>(0x839BB4AC);
  const uint8_t* config = ctx->TranslateVirtual<const uint8_t*>(0x839BBA08);
  XELOGI(
      "[Milo] natal state: disabled={} reqOpen={} reqClose={} okay={} "
      "reopen={:08X} threadRunning={} reqNuiHands={} skelTracking={} "
      "selectedSkeleton={:08X} nuiOpen={} | config: skel={} hands={} "
      "trackActive={} image={} depth={} imgType={} speech={} audio={} "
      "flag25={}",
      natal[0x00], natal[0x01], natal[0x02], natal[0x03],
      xe::load_and_swap<uint32_t>(natal + 0x0C), natal[0x10], natal[0x11],
      natal[0x13], xe::load_and_swap<uint32_t>(natal + 0x2C), natal[0x36],
      config[0x10], config[0x12], config[0x13], config[0x14], config[0x15],
      config[0x16], config[0x17], config[0x24], config[0x25]);

  // Cursor pipeline: CNatalInterface::s_Skeleton.valid, smoothed hands,
  // joystick override, cursor container -> CursorInfo[0] -> sticky cursor.
  auto f32 = [&](uint32_t address) {
    uint32_t bits = Load32(ctx, address);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
  };
  const uint8_t* rhand = ctx->TranslateVirtual<const uint8_t*>(0x839BBA40);
  const uint8_t* lhand = ctx->TranslateVirtual<const uint8_t*>(0x839BB5C0);
  uint32_t container = Load32(ctx, 0x8363D68C);
  uint32_t cursor_enabled = 0, sticky = 0;
  float conf = 0.0f, px = 0.0f, py = 0.0f, pz = 0.0f;
  if (container) {
    cursor_enabled = *ctx->TranslateVirtual<uint8_t*>(container + 0x260);
    sticky = Load32(ctx, container + 0x260 + 0x100);
    if (sticky) {
      conf = f32(sticky + 0x58);
      px = f32(sticky + 0x30);
      py = f32(sticky + 0x34);
      pz = f32(sticky + 0x38);
    }
  }
  // What our own input layer would hand the title for user 0.
  hid::X_INPUT_STATE pad = {};
  auto* input_system = ctx->kernel_state->emulator()->input_system();
  X_RESULT pad_result;
  {
    auto input_lock = input_system->lock();
    pad_result = input_system->GetState(0, hid::X_INPUT_FLAG_GAMEPAD, &pad);
  }
  XELOGI("[Milo] pad0: result={:08X} buttons={:04X} lx={} ly={} rx={} ry={} "
         "lt={} rt={} packet={}",
         pad_result, static_cast<uint16_t>(pad.gamepad.buttons),
         static_cast<int16_t>(pad.gamepad.thumb_lx),
         static_cast<int16_t>(pad.gamepad.thumb_ly),
         static_cast<int16_t>(pad.gamepad.thumb_rx),
         static_cast<int16_t>(pad.gamepad.thumb_ry), pad.gamepad.left_trigger,
         pad.gamepad.right_trigger, static_cast<uint32_t>(pad.packet_number));
  XELOGI(
      "[Milo] cursor: skelValid={} rhand[0..15]={:02X}{:02X}{:02X}{:02X} "
      "{:02X}{:02X}{:02X}{:02X} lhand[0..3]={:02X}{:02X}{:02X}{:02X} "
      "joystickOverride={} container={:08X} cursorEnabled={} sticky={:08X} "
      "conf={:.3f} pos=({:.1f},{:.1f},{:.1f})",
      *ctx->TranslateVirtual<uint8_t*>(0x836480C8), rhand[0], rhand[1],
      rhand[2], rhand[3], rhand[4], rhand[5], rhand[6], rhand[7], lhand[0],
      lhand[1], lhand[2], lhand[3],
      *ctx->TranslateVirtual<uint8_t*>(0x83649FDC), container, cursor_enabled,
      sticky, conf, px, py, pz);
}

// The shared host sensor (xe::nui::NuiSystem, --nui) when it is enabled and
// able to deliver frames. Milo links the NUI runtime statically and its own
// NuiInitialize never reaches the host, so the sensor is started here on first
// use. Returns nullptr when the shared sensor is off or has no source, in
// which case the gamepad-driven VirtualKinect below is used instead.
nui::NuiSystem* SharedNui(KernelState* kernel_state) {
  auto* system = kernel_state->emulator()->nui_system();
  if (!system || !system->is_enabled() || !system->is_device_present()) {
    return nullptr;
  }
  if (!system->initialized()) {
    uint32_t result = system->Initialize(nui::kInitDepthAndPlayerIndex |
                                         nui::kInitSkeleton);
    if (result != nui::kNuiOk && result != nui::kNuiErrorAlreadyInitialized) {
      XELOGW("Project Milo: shared NUI sensor failed to start ({:08X})",
             result);
      return nullptr;
    }
    system->EnableSkeletonTracking(0);
    XELOGI("Project Milo: using the shared NUI sensor for hand tracking");
  }
  return system;
}

// Writes a host skeleton frame into the guest's NUI_SKELETON_FRAME. The host
// structure already uses the sensor's coordinate convention (metres), so the
// fields map across one to one; only the byte order and the timestamp unit
// (100 ns) change.
void WriteGuestSkeletonFrame(uint8_t* frame, const nui::SkeletonFrame& src) {
  const VirtualKinect::Layout l;
  std::memset(frame, 0, l.frame_size);
  xe::store_and_swap<int64_t>(frame + l.frame_timestamp,
                              src.timestamp_us * 10);
  xe::store_and_swap<uint32_t>(frame + l.frame_number, src.frame_number);
  xe::store_and_swap<uint32_t>(frame + l.frame_flags, src.flags);
  auto store_vec = [](uint8_t* p, const nui::Vec4& v) {
    auto f32 = [](float value) {
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      return bits;
    };
    xe::store_and_swap<uint32_t>(p + 0, f32(v.x));
    xe::store_and_swap<uint32_t>(p + 4, f32(v.y));
    xe::store_and_swap<uint32_t>(p + 8, f32(v.z));
    xe::store_and_swap<uint32_t>(p + 12, f32(v.w));
  };
  store_vec(frame + l.frame_floor_clip_plane, src.floor_clip_plane);
  store_vec(frame + l.frame_normal_to_gravity, src.normal_to_gravity);
  const uint32_t count =
      std::min<uint32_t>(l.skeleton_count,
                         static_cast<uint32_t>(src.skeletons.size()));
  for (uint32_t i = 0; i < count; ++i) {
    const nui::Skeleton& skeleton = src.skeletons[i];
    uint8_t* dst = frame + l.frame_skeletons + i * l.skeleton_size;
    xe::store_and_swap<uint32_t>(dst + l.skeleton_tracking_state,
                                 static_cast<uint32_t>(skeleton.state));
    xe::store_and_swap<uint32_t>(dst + l.skeleton_tracking_id,
                                 skeleton.tracking_id);
    xe::store_and_swap<uint32_t>(dst + l.skeleton_enrollment_index,
                                 skeleton.enrollment_index);
    xe::store_and_swap<uint32_t>(dst + l.skeleton_user_index,
                                 skeleton.user_index);
    store_vec(dst + l.skeleton_position, skeleton.position);
    for (uint32_t j = 0; j < nui::kJointCount; ++j) {
      store_vec(dst + l.skeleton_joints + j * 16, skeleton.joints[j]);
      xe::store_and_swap<uint32_t>(
          dst + l.skeleton_joint_states + j * 4,
          static_cast<uint32_t>(skeleton.joint_states[j]));
    }
    xe::store_and_swap<uint32_t>(dst + l.skeleton_quality_flags,
                                 skeleton.quality_flags);
  }
}

VirtualKinect& kinect() {
  static VirtualKinect instance;
  return instance;
}

// Guest-memory helpers (all NUI structures are big-endian).
uint32_t Load32(PPCContext* ctx, uint32_t address) {
  return xe::load_and_swap<uint32_t>(ctx->TranslateVirtual<uint8_t*>(address));
}
void Store32(PPCContext* ctx, uint32_t address, uint32_t value) {
  xe::store_and_swap<uint32_t>(ctx->TranslateVirtual<uint8_t*>(address), value);
}
void Store64(PPCContext* ctx, uint32_t address, uint64_t value) {
  xe::store_and_swap<uint64_t>(ctx->TranslateVirtual<uint8_t*>(address), value);
}

// Kernel-style doubly linked LIST_ENTRY {flink, blink} in guest memory.
uint32_t ListRemoveHead(PPCContext* ctx, uint32_t head) {
  uint32_t first = Load32(ctx, head);
  if (!first || first == head) {
    return 0;
  }
  uint32_t next = Load32(ctx, first);
  Store32(ctx, head, next);
  Store32(ctx, next + 4, head);
  return first;
}
void ListInsertTail(PPCContext* ctx, uint32_t head, uint32_t entry) {
  uint32_t last = Load32(ctx, head + 4);
  Store32(ctx, entry, head);
  Store32(ctx, entry + 4, last);
  Store32(ctx, last, entry);
  Store32(ctx, head + 4, entry);
}

// Layout of the runtime's image stream object (created by NuiImageStreamOpen
// with the title-defined NuiObjectType) and of its frame entries. The handle
// the title holds resolves to this object through the kernel object table.
constexpr uint32_t kStreamEntryCount = 0x18;   // u32 = dwFrameLimit + 2
constexpr uint32_t kStreamEntries = 0x1C;      // entry array pointer
constexpr uint32_t kStreamFreeList = 0x24;     // LIST_ENTRY head
constexpr uint32_t kStreamHeldCount = 0x2C;    // frames held by the title
constexpr uint32_t kStreamHeldList = 0x30;     // LIST_ENTRY head
constexpr uint32_t kEntrySize = 0x68;
constexpr uint32_t kEntryFrame = 0x08;         // NUI_IMAGE_FRAME starts here
constexpr uint32_t kEntryTimestamp = 0x08;     // s64
constexpr uint32_t kEntryFrameNumber = 0x10;   // u32
constexpr uint32_t kEntryImageType = 0x14;     // u32 (0/2 depth, 1 colour)
constexpr uint32_t kEntryPixels = 0x5C;        // pixel buffer pointer
constexpr uint32_t kEntryState = 0x60;         // 0 free, 1 held by title
constexpr uint32_t kDepthBytes = 320 * 240 * 2;
constexpr uint32_t kHResultTooManyFrames = 0x83010004;

// One synthetic stream: paces frames at the sensor rate.
struct StreamClock {
  std::chrono::steady_clock::time_point last;
  uint32_t frame_number = 0;
  bool Due() {
    auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::microseconds(33333)) {
      return false;
    }
    last = now;
    ++frame_number;
    return true;
  }
};

// Delivers an empty (no reading) depth frame through the stream's own entry
// pool so that the title's NuiImageStreamReleaseFrame keeps working
// unmodified. Returns the HRESULT to hand back to the title.
uint32_t DeliverDepthFrame(PPCContext* ctx, uint32_t stream_object,
                           uint32_t out_frame_ptr, StreamClock& clock) {
  uint32_t count = Load32(ctx, stream_object + kStreamEntryCount);
  uint32_t held = Load32(ctx, stream_object + kStreamHeldCount);
  if (count < 3 || held + 2 >= count) {
    return kHResultTooManyFrames;
  }
  uint32_t entry = ListRemoveHead(ctx, stream_object + kStreamFreeList);
  if (!entry) {
    return kHResultPending;
  }
  uint32_t pixels = Load32(ctx, entry + kEntryPixels);
  if (pixels) {
    std::memset(ctx->TranslateVirtual<uint8_t*>(pixels), 0, kDepthBytes);
  }
  auto now = std::chrono::steady_clock::now();
  Store64(ctx, entry + kEntryTimestamp,
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              now.time_since_epoch())
                  .count() /
              100);
  Store32(ctx, entry + kEntryFrameNumber, clock.frame_number);
  Store32(ctx, entry + kEntryState, 1);
  ListInsertTail(ctx, stream_object + kStreamHeldList, entry);
  Store32(ctx, stream_object + kStreamHeldCount, held + 1);
  Store32(ctx, out_frame_ptr, entry + kEntryFrame);
  return 0;
}

// HRESULT NuiImageStreamGetNextFrame(HANDLE stream, DWORD timeout_ms,
//                                    const NUI_IMAGE_FRAME** frame)
void NuiImageStreamGetNextFrame(PPCContext* ctx, KernelState* kernel_state) {
  uint32_t stream = static_cast<uint32_t>(ctx->r[3]);
  uint32_t timeout_ms = static_cast<uint32_t>(ctx->r[4]);
  uint32_t frame_ptr = static_cast<uint32_t>(ctx->r[5]);
  if (!stream || !frame_ptr) {
    ctx->r[3] = kHResultInvalidArg;
    return;
  }
  DumpNatalState(ctx);
  if (cvars::milo_disable_dof) {
    // NEngine::CPostProcessor_DepthOfField::s_nearBlurOn / s_farBlurOn.
    *ctx->TranslateVirtual<uint8_t*>(0x83678908) = 0;
    *ctx->TranslateVirtual<uint8_t*>(0x83678909) = 0;
  }

  // Resolve the handle to the runtime's stream object.
  uint32_t stream_object = 0;
  if (auto object = kernel_state->object_table()->LookupObject<XGuestObject>(
          stream)) {
    stream_object = object->guest_object();
  }
  if (!stream_object) {
    ctx->r[3] = WaitNoFrame(timeout_ms);
    return;
  }
  uint32_t entries = Load32(ctx, stream_object + kStreamEntries);
  uint32_t image_type = entries ? Load32(ctx, entries + kEntryImageType) : 1;
  const bool is_depth = image_type != 1;
  if (!is_depth) {
    // Colour frames are not synthesised (nothing in the game needs them).
    ctx->r[3] = WaitNoFrame(timeout_ms);
    return;
  }

  static StreamClock depth_clock;
  if (!depth_clock.Due()) {
    uint32_t sleep_ms =
        timeout_ms == 0xFFFFFFFF ? 33u : std::min(timeout_ms, 33u);
    if (sleep_ms) {
      xe::threading::Sleep(std::chrono::milliseconds(sleep_ms));
    }
    if (!depth_clock.Due()) {
      ctx->r[3] = kHResultPending;
      return;
    }
  }
  uint32_t result = DeliverDepthFrame(ctx, stream_object, frame_ptr, depth_clock);
  static bool logged_first = false;
  if (!logged_first && result == 0) {
    logged_first = true;
    XELOGI("Project Milo: first virtual Kinect depth frame delivered");
  }
  ctx->r[3] = result;
}

// The seated "NUI Hands" pipeline (spock library) derives hand positions from
// the depth image in four steps driven by the title's render and hands
// threads. Steps 1-3 (CPU pre-processing, GPU passes) are replaced by no-ops
// and step 4, which writes the resulting NUI_SKELETON_FRAME, is fed from the
// virtual Kinect instead.
void NuiHandsNoOp(PPCContext* ctx, KernelState* kernel_state) { ctx->r[3] = 0; }

// HRESULT NuiHands4_CpuPostProc(NUI_SKELETON_FRAME* frame)
void NuiHands4CpuPostProc(PPCContext* ctx, KernelState* kernel_state) {
  uint32_t frame_ptr = static_cast<uint32_t>(ctx->r[3]);
  if (!frame_ptr) {
    ctx->r[3] = kHResultInvalidArg;
    return;
  }
  uint8_t* guest_frame = ctx->TranslateVirtual<uint8_t*>(frame_ptr);
  bool from_sensor = false;
  uint32_t tracked_bodies = 0;
  if (auto* system = SharedNui(kernel_state)) {
    nui::SkeletonFrame frame;
    if (system->GetNextSkeletonFrame(0, 0, &frame)) {
      WriteGuestSkeletonFrame(guest_frame, frame);
      from_sensor = true;
      for (const auto& skeleton : frame.skeletons) {
        if (skeleton.state == nui::SkeletonState::kTracked) {
          ++tracked_bodies;
        }
      }
    }
  }
  if (!from_sensor) {
    auto& vk = kinect();
    vk.Update(kernel_state->emulator()->input_system());
    vk.NewFrameDue();  // advances the frame counter
    vk.WriteSkeletonFrame(guest_frame);
  }
  // Report which source is driving the hands, and how many people the sensor
  // sees, whenever either changes - that is what tells you whether the camera
  // has found you.
  {
    static int last_source = -1;
    static uint32_t last_bodies = 0xFFFFFFFF;
    if (static_cast<int>(from_sensor) != last_source ||
        tracked_bodies != last_bodies) {
      last_source = static_cast<int>(from_sensor);
      last_bodies = tracked_bodies;
      if (from_sensor) {
        XELOGI("Project Milo: hands from the shared NUI sensor, {} person(s) "
               "tracked",
               tracked_bodies);
      } else {
        XELOGI("Project Milo: hands from the gamepad (no sensor frame yet)");
      }
    }
  }
  ctx->r[3] = 0;
}

// HRESULT NuiSkeletonGetNextFrame(DWORD timeout_ms, NUI_SKELETON_FRAME* frame)
void NuiSkeletonGetNextFrame(PPCContext* ctx, KernelState* kernel_state) {
  uint32_t timeout_ms = static_cast<uint32_t>(ctx->r[3]);
  uint32_t frame_ptr = static_cast<uint32_t>(ctx->r[4]);
  if (!frame_ptr) {
    ctx->r[3] = kHResultInvalidArg;
    return;
  }
  if (auto* system = SharedNui(kernel_state)) {
    // The shared sensor paces frames itself; block like the runtime does.
    static std::atomic<uint32_t> last_frame_number{0};
    nui::SkeletonFrame frame;
    uint32_t wait_ms = timeout_ms == 0xFFFFFFFF ? 1000u : timeout_ms;
    if (!system->GetNextSkeletonFrame(last_frame_number.load(), wait_ms,
                                      &frame)) {
      ctx->r[3] = kHResultPending;
      return;
    }
    last_frame_number.store(frame.frame_number);
    WriteGuestSkeletonFrame(ctx->TranslateVirtual<uint8_t*>(frame_ptr), frame);
    ctx->r[3] = 0;
    return;
  }
  auto& vk = kinect();
  vk.Update(kernel_state->emulator()->input_system());
  if (!vk.NewFrameDue()) {
    // Behave like the runtime: wait up to the timeout for the next 30 Hz
    // frame, then report E_PENDING if it is still not due.
    uint32_t sleep_ms =
        timeout_ms == 0xFFFFFFFF ? 33u : std::min(timeout_ms, 33u);
    if (sleep_ms) {
      xe::threading::Sleep(std::chrono::milliseconds(sleep_ms));
    }
    if (!vk.NewFrameDue()) {
      ctx->r[3] = kHResultPending;
      return;
    }
  }
  vk.WriteSkeletonFrame(ctx->TranslateVirtual<uint8_t*>(frame_ptr));
  static bool logged_first = false;
  if (!logged_first) {
    logged_first = true;
    XELOGI("Project Milo: first virtual Kinect skeleton frame delivered");
  }
  ctx->r[3] = 0;
}

// void LHDebug::LogMessage(const char* message, const char* file, uint flags)
// The title's own trace output normally goes to an on-screen console and a
// devkit log file; mirror it into our log so the game's state is visible.
void LHDebugLogMessage(PPCContext* ctx, KernelState* kernel_state) {
  uint32_t message_ptr = static_cast<uint32_t>(ctx->r[3]);
  if (message_ptr) {
    std::string_view message(ctx->TranslateVirtual<const char*>(message_ptr));
    while (!message.empty() &&
           (message.back() == '\n' || message.back() == '\r' ||
            message.back() == ' ')) {
      message.remove_suffix(1);
    }
    XELOGI("[Milo] {}", message);
  }
  ctx->r[3] = 0;
}

struct Hook {
  const char* name;
  uint32_t address;
  std::vector<uint32_t> expected_words;
  cpu::GuestFunction::ExternHandler handler;
};

struct BuildHooks {
  const char* module_name;
  std::vector<Hook> hooks;
};

// Addresses come from the linker map / PDB shipped with each build; the
// expected words are the function prologue and protect against mismatches.
const BuildHooks kBuilds[] = {
    {"miloReleaseLIB.xex",
     {
         {"NuiImageStreamGetNextFrame", 0x82C64DA0,
          {0x7D8802A6, 0x4B97DF55}, NuiImageStreamGetNextFrame},
         {"NuiSkeletonGetNextFrame", 0x82C66C10,
          {0x7D8802A6, 0x4B97C0E1}, NuiSkeletonGetNextFrame},
         {"LHDebug::LogMessage", 0x822E37E0, {0x7D8802A6, 0x482FF519},
          LHDebugLogMessage},
         {"NuiHands1_CpuPreProc", 0x822D0178, {0x7D8802A6, 0x9181FFF8},
          NuiHandsNoOp},
         {"NuiHands2_KickOffGPU", 0x822D0348, {0x7D8802A6, 0x9181FFF8},
          NuiHandsNoOp},
         {"NuiHands3_FinishGPU", 0x822D0388, {0x7D8802A6, 0x9181FFF8},
          NuiHandsNoOp},
         {"NuiHands4_CpuPostProc", 0x822D03D0, {0x7D8802A6, 0x48312925},
          NuiHands4CpuPostProc},
     }},
};

}  // namespace

void Install(KernelState* kernel_state, UserModule* module) {
  if (!cvars::virtual_kinect) {
    return;
  }
  auto xex = module->xex_module();
  const std::string_view module_name = module->name();
  // Module names may or may not carry the .xex extension depending on how
  // the title was launched; compare case-insensitively without it.
  auto strip = [](std::string_view s) {
    if (s.size() > 4) {
      std::string_view ext = s.substr(s.size() - 4);
      if ((ext[0] == '.') && (ext[1] == 'x' || ext[1] == 'X') &&
          (ext[2] == 'e' || ext[2] == 'E') && (ext[3] == 'x' || ext[3] == 'X')) {
        return s.substr(0, s.size() - 4);
      }
    }
    return s;
  };
  auto same_name = [&](std::string_view a, std::string_view b) {
    a = strip(a);
    b = strip(b);
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
             return std::tolower(static_cast<unsigned char>(x)) ==
                    std::tolower(static_cast<unsigned char>(y));
           });
  };
  for (const auto& build : kBuilds) {
    if (!same_name(module_name, build.module_name)) {
      continue;
    }
    XELOGI("Project Milo: installing virtual Kinect hooks for {}", module_name);
    for (const auto& hook : build.hooks) {
      xex->InstallExternHook(hook.address, hook.name, hook.handler,
                             hook.expected_words);
    }
    return;
  }
  XELOGW("Project Milo: no virtual Kinect hooks known for module {}",
         module_name);
}

}  // namespace milo
}  // namespace hooks
}  // namespace kernel
}  // namespace xe
