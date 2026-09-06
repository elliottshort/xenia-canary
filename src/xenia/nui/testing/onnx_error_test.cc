/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstdint>
#include <string>

#include "third_party/catch/include/catch.hpp"
#include "xenia/nui/onnx_pose_estimator.h"
#include "xenia/nui/onnx_runtime.h"

namespace xe {
namespace nui {
namespace test {

namespace {

using Action = InferenceRecovery::Action;
using State = InferenceRecovery::State;

constexpr uint64_t kSecond = 1000000;

// Drives the machine from healthy to "waiting for the first GPU rebuild".
InferenceRecovery LostDevice(uint64_t now_us) {
  InferenceRecovery recovery;
  REQUIRE(recovery.RecordFailure(now_us, /*device_lost=*/true) ==
          Action::kWait);
  REQUIRE(recovery.state() == State::kRecovering);
  return recovery;
}

}  // namespace

// ---------------------------------------------------------------------------
// OnnxErrorIsDeviceLost
// ---------------------------------------------------------------------------

TEST_CASE("OnnxErrorIsDeviceLost recognizes DirectML lost-device messages",
          "[nui][onnx]") {
  // Wording seen from the DirectML execution provider and from ORT itself
  // when a TDR takes the D3D12 device down mid-Run.
  CHECK(OnnxErrorIsDeviceLost(
      "Run: Non-zero status code returned while running FusedConv node. "
      "Name:'Conv_0' Status Message: DML_GRAPH_EXECUTE failed with "
      "DXGI_ERROR_DEVICE_HUNG"));
  CHECK(OnnxErrorIsDeviceLost(
      "Run: D3D12 device removed, reason DXGI_ERROR_DEVICE_REMOVED"));
  CHECK(OnnxErrorIsDeviceLost("Run: DXGI_ERROR_DEVICE_RESET"));
  CHECK(OnnxErrorIsDeviceLost(
      "Run: dml execution provider: DXGI_ERROR_DRIVER_INTERNAL_ERROR"));
  // HRESULTs, with and without the 0x prefix and in either case.
  CHECK(OnnxErrorIsDeviceLost("Run: HRESULT failed with 0x887A0006"));
  CHECK(OnnxErrorIsDeviceLost("Run: hr = 0x887a0005"));
  CHECK(OnnxErrorIsDeviceLost("Run: GetDeviceRemovedReason: 887A0007"));
  CHECK(OnnxErrorIsDeviceLost("Run: 887A0020 while flushing the queue"));
  // Plain-English wording.
  CHECK(OnnxErrorIsDeviceLost(
      "Run: the GPU device instance has been suspended, device removed"));
  CHECK(OnnxErrorIsDeviceLost("Run: The device was removed (TDR)"));
  CHECK(OnnxErrorIsDeviceLost("Run: DML device lost"));
}

TEST_CASE("OnnxErrorIsDeviceLost ignores ordinary inference errors",
          "[nui][onnx]") {
  CHECK_FALSE(OnnxErrorIsDeviceLost(""));
  // Shape / input mismatches.
  CHECK_FALSE(OnnxErrorIsDeviceLost(
      "Run: Invalid rank for input: input_1 Got: 3 Expected: 4 Please fix "
      "either the inputs or the model."));
  CHECK_FALSE(OnnxErrorIsDeviceLost(
      "CreateTensorWithDataAsOrtValue: input 'input_1' has no data or an "
      "invalid shape [1,-1,256,3]"));
  // Missing files and bad models.
  CHECK_FALSE(OnnxErrorIsDeviceLost(
      "CreateSession: Load model from nui/models/pose_landmark_full.onnx "
      "failed: No such file or directory"));
  CHECK_FALSE(OnnxErrorIsDeviceLost(
      "Run: Non-zero status code returned while running Concat node. "
      "Name:'Concat_2' Status Message: Concat: input tensors must have the "
      "same shape"));
  // A device that never came up is not a device that was lost mid-run, but
  // it must not be mistaken for one either.
  CHECK_FALSE(OnnxErrorIsDeviceLost(
      "DirectML execution provider requested but unavailable"));
  CHECK_FALSE(
      OnnxErrorIsDeviceLost("Run: output 'Identity' is not float32 (element "
                            "type 10)"));
}

// ---------------------------------------------------------------------------
// InferenceRecovery
// ---------------------------------------------------------------------------

TEST_CASE("InferenceRecovery stays out of the way while healthy",
          "[nui][onnx]") {
  InferenceRecovery recovery;
  CHECK(recovery.state() == State::kHealthy);
  CHECK(recovery.Poll(0) == Action::kContinue);
  // A handful of ordinary failures is not worth a rebuild.
  for (uint32_t i = 1; i < InferenceRecovery::kFailureThreshold; ++i) {
    CHECK(recovery.RecordFailure(i * kSecond, /*device_lost=*/false) ==
          Action::kContinue);
    CHECK(recovery.state() == State::kHealthy);
  }
  recovery.RecordSuccess();
  CHECK(recovery.consecutive_failures() == 0);
  CHECK(recovery.state() == State::kHealthy);
}

TEST_CASE("InferenceRecovery rebuilds after kFailureThreshold failures",
          "[nui][onnx]") {
  InferenceRecovery recovery;
  Action action = Action::kContinue;
  for (uint32_t i = 0; i < InferenceRecovery::kFailureThreshold; ++i) {
    action = recovery.RecordFailure(kSecond, /*device_lost=*/false);
  }
  CHECK(action == Action::kWait);
  CHECK(recovery.state() == State::kRecovering);
  // Same 2 s backoff as a device loss.
  CHECK(recovery.retry_deadline_us() == kSecond + 2 * kSecond);
}

TEST_CASE("InferenceRecovery backs off 2s, 5s then 15s", "[nui][onnx]") {
  const uint64_t t0 = 100 * kSecond;
  InferenceRecovery recovery = LostDevice(t0);

  // First attempt: not before 2 s.
  CHECK(recovery.retry_deadline_us() == t0 + 2 * kSecond);
  CHECK(recovery.Poll(t0) == Action::kWait);
  CHECK(recovery.Poll(t0 + 2 * kSecond - 1) == Action::kWait);
  CHECK(recovery.Poll(t0 + 2 * kSecond) == Action::kRebuildGpu);
  // Further failures while recovering never shorten the wait.
  CHECK(recovery.RecordFailure(t0 + kSecond, /*device_lost=*/true) ==
        Action::kWait);
  CHECK(recovery.retry_deadline_us() == t0 + 2 * kSecond);

  // Attempt 1 fails -> 5 s.
  const uint64_t t1 = t0 + 2 * kSecond;
  recovery.RecordRebuildResult(/*success=*/false, /*on_cpu=*/false, t1);
  CHECK(recovery.gpu_attempts() == 1);
  CHECK(recovery.Poll(t1 + 4 * kSecond) == Action::kWait);
  CHECK(recovery.Poll(t1 + 5 * kSecond) == Action::kRebuildGpu);

  // Attempt 2 fails -> 15 s.
  const uint64_t t2 = t1 + 5 * kSecond;
  recovery.RecordRebuildResult(/*success=*/false, /*on_cpu=*/false, t2);
  CHECK(recovery.gpu_attempts() == 2);
  CHECK(recovery.Poll(t2 + 14 * kSecond) == Action::kWait);
  CHECK(recovery.Poll(t2 + 15 * kSecond) == Action::kRebuildGpu);

  // Attempt 3 fails: the GPU attempts are used up, the CPU rebuild follows
  // immediately.
  const uint64_t t3 = t2 + 15 * kSecond;
  recovery.RecordRebuildResult(/*success=*/false, /*on_cpu=*/false, t3);
  CHECK(recovery.gpu_attempts() == InferenceRecovery::kMaxGpuAttempts);
  CHECK(recovery.Poll(t3) == Action::kRebuildCpu);
}

TEST_CASE("InferenceRecovery goes back to healthy when the GPU comes back",
          "[nui][onnx]") {
  const uint64_t t0 = 10 * kSecond;
  InferenceRecovery recovery = LostDevice(t0);
  const uint64_t t1 = t0 + 2 * kSecond;
  REQUIRE(recovery.Poll(t1) == Action::kRebuildGpu);
  recovery.RecordRebuildResult(/*success=*/true, /*on_cpu=*/false, t1);
  CHECK(recovery.state() == State::kHealthy);
  CHECK(recovery.Poll(t1) == Action::kContinue);
  // A later, unrelated loss gets the full set of attempts again.
  CHECK(recovery.gpu_attempts() == 0);
  const uint64_t t2 = t1 + 600 * kSecond;
  CHECK(recovery.RecordFailure(t2, /*device_lost=*/true) == Action::kWait);
  CHECK(recovery.retry_deadline_us() == t2 + 2 * kSecond);
}

TEST_CASE("InferenceRecovery falls back to the CPU and stays there",
          "[nui][onnx]") {
  uint64_t now = 0;
  InferenceRecovery recovery = LostDevice(now);
  for (uint32_t attempt = 0; attempt < InferenceRecovery::kMaxGpuAttempts;
       ++attempt) {
    now += 20 * kSecond;
    REQUIRE(recovery.Poll(now) == Action::kRebuildGpu);
    recovery.RecordRebuildResult(/*success=*/false, /*on_cpu=*/false, now);
  }
  REQUIRE(recovery.Poll(now) == Action::kRebuildCpu);
  recovery.RecordRebuildResult(/*success=*/true, /*on_cpu=*/true, now);
  CHECK(recovery.state() == State::kDegradedCpu);
  CHECK(recovery.Poll(now) == Action::kContinue);

  // Occasional failures on the CPU do not start the GPU dance again.
  now += kSecond;
  CHECK(recovery.RecordFailure(now, /*device_lost=*/false) ==
        Action::kContinue);
  recovery.RecordSuccess();
  CHECK(recovery.state() == State::kDegradedCpu);

  // But a solid run of them ends the run.
  Action action = Action::kContinue;
  for (uint32_t i = 0; i < InferenceRecovery::kFailureThreshold; ++i) {
    now += kSecond;
    action = recovery.RecordFailure(now, /*device_lost=*/false);
  }
  CHECK(action == Action::kGiveUp);
  CHECK(recovery.state() == State::kFailed);
  CHECK(recovery.Poll(now + 3600 * kSecond) == Action::kGiveUp);
  CHECK(recovery.RecordFailure(now, /*device_lost=*/true) == Action::kGiveUp);
}

TEST_CASE("InferenceRecovery gives up when the CPU rebuild fails too",
          "[nui][onnx]") {
  uint64_t now = 5 * kSecond;
  InferenceRecovery recovery = LostDevice(now);
  for (uint32_t attempt = 0; attempt < InferenceRecovery::kMaxGpuAttempts;
       ++attempt) {
    now += 20 * kSecond;
    REQUIRE(recovery.Poll(now) == Action::kRebuildGpu);
    recovery.RecordRebuildResult(/*success=*/false, /*on_cpu=*/false, now);
  }
  REQUIRE(recovery.Poll(now) == Action::kRebuildCpu);
  recovery.RecordRebuildResult(/*success=*/false, /*on_cpu=*/true, now);
  CHECK(recovery.state() == State::kFailed);
  CHECK(recovery.Poll(now) == Action::kGiveUp);
}

}  // namespace test
}  // namespace nui
}  // namespace xe
