/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/base/logging.h"
#include "xenia/cpu/processor.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/kernel/xboxkrnl/xboxkrnl_private.h"
#include "xenia/kernel/xthread.h"
#include "xenia/xbox.h"

// Kinect (NUI) sensor device requests.
//
// The Kinect sensor is driven by a runtime library that is statically linked
// into titles (nuiapi/nuiruntime). That library talks to the hardware only
// through three kernel imports, each taking a single request block:
//   PsCamDeviceRequest   - the PrimeSense depth/colour camera
//   DetroitDeviceRequest - the auxiliary microcontroller (tilt motor,
//                          accelerometer, status)
//   McaDeviceRequest     - the microphone array
// Request layouts below were reverse-engineered from the runtime shipped with
// Project Milo (XDK 11427) and match the later retail Kinect titles.
//
// Without a sensor attached, the runtime expects STATUS_DEVICE_NOT_CONNECTED
// (or STATUS_DEVICE_NOT_READY) from the "query state" requests: it then keeps
// polling every 30 ms from its worker thread, NuiInitialize still succeeds and
// title stream/skeleton reads time out with "no data". Any other return value
// (including STATUS_SUCCESS and STATUS_NOT_IMPLEMENTED) is interpreted as
// "device present" and makes the runtime parse uninitialised buffers.

namespace xe {
namespace kernel {
namespace xboxkrnl {

namespace {

constexpr X_STATUS kStatusDeviceNotConnected = 0xC000009D;

// Common prefix of every camera request. Only |opcode| is valid for every
// opcode; the remaining fields exist for control (1, 2, 6, 7) and transfer
// (5) requests. Opcode 0 requests are only 8 bytes long.
struct X_NUICAM_REQUEST {
  xe::be<uint32_t> opcode;       // 0x00
  xe::be<uint32_t> state;        // 0x04 opcode 0: out device state
  xe::be<uint32_t> reserved[2];  // 0x08
  xe::be<uint32_t> context;      // 0x10 completion context
  xe::be<uint32_t> callback;     // 0x14 completion callback
  xe::be<uint32_t> param;        // 0x18 control: parameter; transfer: type
  xe::be<uint32_t> buffer;       // 0x1C transfer: destination buffer
  xe::be<uint32_t> length;       // 0x20 transfer: buffer length
};

enum NuiCamOpcode : uint32_t {
  kNuiCamQueryState = 0x00,
  kNuiCamOpen = 0x01,
  kNuiCamClose = 0x02,
  kNuiCamTransfer = 0x05,
  kNuiCamControl6 = 0x06,
  kNuiCamControl7 = 0x07,
  kNuiCamRegistrationParamsV1 = 0x0B,
  kNuiCamGetVersion = 0x0D,
  kNuiCamRegistrationParamsV2 = 0x0E,
  kNuiCamUpdateBlanking = 0x0F,
  kNuiCamStart = 0x12,
  kNuiCamGetVBlankInfo = 0x19,
};

bool IsNuiCamControlOpcode(uint32_t opcode) {
  return opcode == kNuiCamOpen || opcode == kNuiCamClose ||
         opcode == kNuiCamControl6 || opcode == kNuiCamControl7;
}

// The runtime's aux (Detroit) request block.
struct X_DETROIT_CONTROL_REQUEST {
  xe::be<uint32_t> reserved0;   // 0x00
  xe::be<uint32_t> reserved1;   // 0x04
  xe::be<uint32_t> code;        // 0x08 0x000F0410 status query, 0x00033148
                                //      set elevation
  xe::be<uint32_t> reserved2;   // 0x0C
  xe::be<uint32_t> context;     // 0x10
  xe::be<uint32_t> callback;    // 0x14 void(req*, NTSTATUS) or 0
  xe::be<int16_t> param;        // 0x18 elevation angle in degrees
  xe::be<uint16_t> param2;      // 0x1A
  xe::be<uint16_t> out_length;  // 0x1C
  xe::be<uint16_t> pad;         // 0x1E
  xe::be<uint32_t> out_buffer;  // 0x20
};
static_assert_size(X_DETROIT_CONTROL_REQUEST, 0x24);

}  // namespace

// NTSTATUS PsCamDeviceRequest(_NUICAM_REQUEST* request)
dword_result_t PsCamDeviceRequest_entry(pointer_t<X_NUICAM_REQUEST> request,
                                        const ppc_context_t& ctx) {
  if (!request) {
    return X_STATUS_INVALID_PARAMETER;
  }

  const uint32_t opcode = request->opcode;
  const X_STATUS status = kStatusDeviceNotConnected;

  if (IsNuiCamControlOpcode(opcode)) {
    // Control requests are completed through their callback: the runtime
    // ignores our return value and waits (without timeout) on an event that
    // only the callback signals, so it must always be invoked. Completing
    // synchronously from inside the call is fine.
    const uint32_t callback = request->callback;
    if (callback) {
      uint64_t args[] = {request.guest_address(), status};
      kernel_state()->processor()->Execute(ctx->thread_state, callback, args,
                                           xe::countof(args));
    }
    return status;
  }

  // Transfer requests (opcode 5) must NOT be completed through the callback
  // when rejected; the runtime unwinds its bookkeeping itself when anything
  // other than STATUS_PENDING/STATUS_SUCCESS is returned. All other opcodes
  // are synchronous queries whose outputs we leave untouched.
  if (opcode != kNuiCamQueryState) {
    XELOGD("PsCamDeviceRequest: opcode {:#x} rejected, no camera attached",
           opcode);
  }
  return status;
}
DECLARE_XBOXKRNL_EXPORT2(PsCamDeviceRequest, kNone, kStub, kHighFrequency);

// NTSTATUS DetroitDeviceRequest(DETROIT_CONTROL_REQUEST* request)
dword_result_t DetroitDeviceRequest_entry(
    pointer_t<X_DETROIT_CONTROL_REQUEST> request) {
  if (!request) {
    return X_STATUS_INVALID_PARAMETER;
  }
  // The runtime treats a non-pending failure as "completed synchronously"
  // and clears its own pending state, so the callback must not be invoked.
  // Do not touch the output buffer either: it is parsed only on success.
  XELOGD("DetroitDeviceRequest: code {:#x} rejected, no sensor attached",
         static_cast<uint32_t>(request->code));
  return kStatusDeviceNotConnected;
}
DECLARE_XBOXKRNL_EXPORT2(DetroitDeviceRequest, kNone, kStub, kHighFrequency);

// NTSTATUS McaDeviceRequest(void* request)
dword_result_t McaDeviceRequest_entry(lpvoid_t request) {
  if (!request) {
    return X_STATUS_INVALID_PARAMETER;
  }
  return kStatusDeviceNotConnected;
}
DECLARE_XBOXKRNL_EXPORT1(McaDeviceRequest, kNone, kStub);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

DECLARE_XBOXKRNL_EMPTY_REGISTER_EXPORTS(Nui);
