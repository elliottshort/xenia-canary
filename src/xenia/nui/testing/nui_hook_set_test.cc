/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <string>
#include <vector>

#include "third_party/catch/include/catch.hpp"
#include "xenia/nui/nui_hook_sets.h"

namespace xe {
namespace nui {
namespace test {

namespace {

NuiHookCandidate Ours(const std::string& name, NuiHookSet set,
                      uint32_t address) {
  NuiHookCandidate candidate;
  candidate.name = name;
  candidate.set = set;
  candidate.address = address;
  candidate.resolved = true;
  return candidate;
}

NuiHookCandidate Theirs(const std::string& name, NuiHookSet set,
                        uint32_t address) {
  NuiHookCandidate candidate = Ours(name, set, address);
  candidate.already_hooked = true;
  return candidate;
}

NuiHookCandidate NotFound(const std::string& name, NuiHookSet set) {
  NuiHookCandidate candidate;
  candidate.name = name;
  candidate.set = set;
  return candidate;
}

// The four sets of the built-in signature table, all resolved and ours.
std::vector<NuiHookCandidate> FullTable() {
  return {
      Ours("NuiInitialize", NuiHookSet::kLifecycle, 0x82C60000),
      Ours("NuiShutdown", NuiHookSet::kLifecycle, 0x82C60100),
      Ours("NuiSkeletonTrackingEnable", NuiHookSet::kSkeleton, 0x82C66000),
      Ours("NuiSkeletonGetNextFrame", NuiHookSet::kSkeleton, 0x82C66C10),
      Ours("NuiImageStreamOpen", NuiHookSet::kImage, 0x82C64000),
      Ours("NuiImageStreamGetNextFrame", NuiHookSet::kImage, 0x82C64DA0),
      Ours("NuiImageStreamReleaseFrame", NuiHookSet::kImage, 0x82C64F00),
      Ours("NuiCameraElevationGetAngle", NuiHookSet::kCamera, 0x82C67000),
  };
}

size_t IndexOf(const std::vector<NuiHookCandidate>& candidates,
               const std::string& name) {
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].name == name) {
      return i;
    }
  }
  return candidates.size();
}

const NuiHookSetDecision* DecisionFor(const NuiHookPlan& plan, NuiHookSet set) {
  for (const auto& decision : plan.decisions) {
    if (decision.set == set) {
      return &decision;
    }
  }
  return nullptr;
}

}  // namespace

TEST_CASE("A runtime nobody else touched is hooked whole", "[nui]") {
  const auto candidates = FullTable();
  const NuiHookPlan plan = PlanNuiHooks(candidates);
  REQUIRE(plan.hooked_count == candidates.size());
  REQUIRE(plan.required_hooked_count == candidates.size());
  REQUIRE(plan.ceded_set_count == 0);
  REQUIRE(plan.missing_set_count == 0);
  REQUIRE(plan.decisions.size() == 4);
  for (size_t i = 0; i < candidates.size(); ++i) {
    REQUIRE(plan.ShouldHook(i));
  }
}

TEST_CASE("One function owned by title hooks cedes its whole set", "[nui]") {
  auto candidates = FullTable();
  // What Project Milo's title hooks do: they own NuiImageStreamGetNextFrame
  // and expect their own stream objects, so NuiImageStreamOpen must not hand
  // out ours.
  candidates[IndexOf(candidates, "NuiImageStreamGetNextFrame")] =
      Theirs("NuiImageStreamGetNextFrame", NuiHookSet::kImage, 0x82C64DA0);
  const NuiHookPlan plan = PlanNuiHooks(candidates);

  REQUIRE(plan.IsSetCeded(NuiHookSet::kImage));
  REQUIRE(plan.ceded_set_count == 1);
  REQUIRE_FALSE(plan.ShouldHook(IndexOf(candidates, "NuiImageStreamOpen")));
  REQUIRE_FALSE(
      plan.ShouldHook(IndexOf(candidates, "NuiImageStreamGetNextFrame")));
  REQUIRE_FALSE(
      plan.ShouldHook(IndexOf(candidates, "NuiImageStreamReleaseFrame")));
  // Every other set is untouched by the cede.
  REQUIRE(plan.ShouldHook(IndexOf(candidates, "NuiInitialize")));
  REQUIRE(plan.ShouldHook(IndexOf(candidates, "NuiSkeletonGetNextFrame")));
  REQUIRE(plan.ShouldHook(IndexOf(candidates, "NuiCameraElevationGetAngle")));
  REQUIRE(plan.required_hooked_count == 5);
  REQUIRE(plan.IsSetHooked(NuiHookSet::kSkeleton));
  REQUIRE_FALSE(plan.IsSetHooked(NuiHookSet::kImage));

  const NuiHookSetDecision* image = DecisionFor(plan, NuiHookSet::kImage);
  REQUIRE(image != nullptr);
  REQUIRE(image->status == NuiHookSetStatus::kCeded);
  REQUIRE(image->blocking_function == "NuiImageStreamGetNextFrame");
  REQUIRE(image->blocking_address == 0x82C64DA0);
  REQUIRE(image->label == std::string("image stream"));
}

TEST_CASE("A missing function only drops its own set", "[nui]") {
  auto candidates = FullTable();
  candidates[IndexOf(candidates, "NuiSkeletonTrackingEnable")] =
      NotFound("NuiSkeletonTrackingEnable", NuiHookSet::kSkeleton);
  const NuiHookPlan plan = PlanNuiHooks(candidates);

  REQUIRE(plan.missing_set_count == 1);
  REQUIRE(plan.ceded_set_count == 0);
  REQUIRE_FALSE(plan.IsSetHooked(NuiHookSet::kSkeleton));
  REQUIRE_FALSE(
      plan.ShouldHook(IndexOf(candidates, "NuiSkeletonGetNextFrame")));
  REQUIRE(plan.ShouldHook(IndexOf(candidates, "NuiImageStreamOpen")));
  REQUIRE(plan.required_hooked_count == 6);

  const NuiHookSetDecision* skeleton = DecisionFor(plan, NuiHookSet::kSkeleton);
  REQUIRE(skeleton != nullptr);
  REQUIRE(skeleton->status == NuiHookSetStatus::kMissing);
  REQUIRE(skeleton->blocking_function == "NuiSkeletonTrackingEnable");
}

TEST_CASE("A foreign owner outranks a missing function in the same set",
          "[nui]") {
  auto candidates = FullTable();
  candidates[IndexOf(candidates, "NuiImageStreamOpen")] =
      NotFound("NuiImageStreamOpen", NuiHookSet::kImage);
  candidates[IndexOf(candidates, "NuiImageStreamGetNextFrame")] =
      Theirs("NuiImageStreamGetNextFrame", NuiHookSet::kImage, 0x82C64DA0);
  const NuiHookPlan plan = PlanNuiHooks(candidates);

  REQUIRE(plan.IsSetCeded(NuiHookSet::kImage));
  REQUIRE(plan.missing_set_count == 0);
  REQUIRE(plan.ceded_set_count == 1);
}

TEST_CASE("Standalone entries are decided one by one", "[nui]") {
  std::vector<NuiHookCandidate> candidates = {
      Ours("NuiImageGetColorPixelCoordinatesFromDepthPixel",
           NuiHookSet::kTransform, 0x82C68000),
      Theirs("NuiSpeechEnable", NuiHookSet::kNone, 0x82C69000),
      NotFound("NuiHandsInitialize", NuiHookSet::kNone),
      Ours("NuiIdentityEnroll", NuiHookSet::kNone, 0x82C6A000),
  };
  candidates[2].required = false;  // Optional tripwire.
  const NuiHookPlan plan = PlanNuiHooks(candidates);

  REQUIRE(plan.decisions.size() == 4);  // The set plus one per standalone.
  REQUIRE(plan.ShouldHook(0));
  REQUIRE_FALSE(plan.ShouldHook(1));
  REQUIRE_FALSE(plan.ShouldHook(2));
  REQUIRE(plan.ShouldHook(3));
  REQUIRE(plan.ceded_set_count == 1);
  // An optional entry that was not found neither hooks nor fails.
  REQUIRE(plan.missing_set_count == 0);
  REQUIRE(plan.decisions[1].label == std::string("NuiSpeechEnable"));
}

TEST_CASE("An optional entry inside a set does not spoil it", "[nui]") {
  auto candidates = FullTable();
  NuiHookCandidate optional_entry =
      NotFound("NuiImageStreamSetImageFrameFlags", NuiHookSet::kImage);
  optional_entry.required = false;
  candidates.push_back(optional_entry);
  const NuiHookPlan plan = PlanNuiHooks(candidates);

  REQUIRE(plan.missing_set_count == 0);
  REQUIRE(plan.IsSetHooked(NuiHookSet::kImage));
  REQUIRE(plan.ShouldHook(IndexOf(candidates, "NuiImageStreamOpen")));
  REQUIRE_FALSE(
      plan.ShouldHook(IndexOf(candidates, "NuiImageStreamSetImageFrameFlags")));
}

TEST_CASE("Only stateful sets count as a served sensor", "[nui]") {
  // A standalone helper owned by title hooks, a pure coordinate transform of
  // ours, and no skeleton set: the title has no Kinect, whatever the raw
  // hooked/ceded counts say.
  std::vector<NuiHookCandidate> candidates = {
      Theirs("NuiHandsGetFrame", NuiHookSet::kNone, 0x82C60000),
      Ours("NuiImageGetColorPixelCoordinatesFromDepthPixel",
           NuiHookSet::kTransform, 0x82C61000),
      NotFound("NuiSkeletonGetNextFrame", NuiHookSet::kSkeleton),
  };
  const NuiHookPlan plan = PlanNuiHooks(candidates);
  CHECK(plan.ceded_set_count == 1);
  CHECK(plan.required_hooked_count == 1);
  CHECK(plan.ceded_stateful_set_count == 0);
  CHECK(plan.hooked_stateful_set_count == 0);
}

TEST_CASE("Stateful sets are counted whoever owns them", "[nui]") {
  std::vector<NuiHookCandidate> candidates = FullTable();
  // The image set goes to a title hook layer, the rest stays ours.
  candidates[IndexOf(candidates, "NuiImageStreamGetNextFrame")].already_hooked =
      true;
  const NuiHookPlan plan = PlanNuiHooks(candidates);
  CHECK(plan.ceded_stateful_set_count == 1);
  // Lifecycle and skeleton; the camera set is not stateful.
  CHECK(plan.hooked_stateful_set_count == 2);
  CHECK(plan.IsSetHooked(NuiHookSet::kLifecycle));
  CHECK(plan.IsSetCeded(NuiHookSet::kImage));
}

TEST_CASE("Nothing left for us when every set is taken or missing", "[nui]") {
  std::vector<NuiHookCandidate> candidates = {
      Theirs("NuiSkeletonGetNextFrame", NuiHookSet::kSkeleton, 0x82C66C10),
      NotFound("NuiInitialize", NuiHookSet::kLifecycle),
  };
  const NuiHookPlan plan = PlanNuiHooks(candidates);
  REQUIRE(plan.required_hooked_count == 0);
  REQUIRE(plan.hooked_count == 0);
  REQUIRE(plan.ceded_set_count == 1);
  REQUIRE(plan.missing_set_count == 1);
}

}  // namespace test
}  // namespace nui
}  // namespace xe
