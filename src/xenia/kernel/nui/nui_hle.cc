/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/nui/nui_hle.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

#include "xenia/base/byte_order.h"
#include "xenia/base/filesystem.h"
#include "xenia/base/logging.h"
#include "xenia/base/string_util.h"
#include "xenia/cpu/xex_module.h"
#include "xenia/emulator.h"
#include "xenia/kernel/kernel_state.h"
#include "xenia/kernel/nui/nui_hle_handlers.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/util/xex2_info.h"
#include "xenia/memory.h"
#include "xenia/nui/nui_flags.h"
#include "xenia/nui/nui_system.h"

namespace xe {
namespace kernel {
namespace nui {

namespace {

constexpr uint32_t kSyscallTrampolineWord = 0x44000042;  // sc 2

NuiHleQuirks g_active_quirks;

struct StaticLibrary {
  std::string name;
  std::string version;
};

std::vector<StaticLibrary> ReadStaticLibraries(cpu::XexModule* xex) {
  std::vector<StaticLibrary> out;
  xex2_opt_static_libraries* libs = nullptr;
  if (!xex->GetOptHeader(XEX_HEADER_STATIC_LIBRARIES, &libs) || !libs) {
    return out;
  }
  const uint32_t count = (libs->size - 4) / 0x10;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& lib = libs->libraries[i];
    StaticLibrary entry;
    entry.name = std::string(lib.name, strnlen(lib.name, sizeof(lib.name)));
    entry.version = fmt::format("{}.{}.{}.{}", uint16_t(lib.version_major),
                                uint16_t(lib.version_minor),
                                uint16_t(lib.version_build),
                                uint16_t(lib.version_qfe));
    out.push_back(std::move(entry));
  }
  return out;
}

std::string StripXexExtension(std::string_view name) {
  std::string out(name);
  if (out.size() > 4) {
    std::string ext = out.substr(out.size() - 4);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".xex") {
      out.resize(out.size() - 4);
    }
  }
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

bool MatchAt(Memory* memory, uint32_t address,
             const NuiFunctionSignature& signature) {
  if (signature.words.empty()) {
    return false;
  }
  const uint8_t* p = memory->TranslateVirtual(address);
  for (size_t i = 0; i < signature.words.size(); ++i) {
    const uint32_t mask =
        i < signature.masks.size() ? signature.masks[i] : 0xFFFFFFFFu;
    const uint32_t word = xe::load_and_swap<uint32_t>(p + i * 4);
    if ((word & mask) != (signature.words[i] & mask)) {
      return false;
    }
  }
  return true;
}

// Searches every code page of the module for a unique match. Returns 0 when
// not found or ambiguous.
uint32_t FindByPattern(Memory* memory, cpu::XexModule* xex,
                       const NuiFunctionSignature& signature,
                       uint32_t* out_match_count) {
  *out_match_count = 0;
  if (signature.words.empty()) {
    return 0;
  }
  uint32_t found = 0;
  const auto* security_info = xex->xex_security_info();
  auto heap = memory->LookupHeap(xex->base_address());
  const uint32_t page_size = heap ? heap->page_size() : 0x1000;
  uint32_t page = 0;
  for (uint32_t i = 0; i < security_info->page_descriptor_count; ++i) {
    xex2_page_descriptor desc;
    desc.value = xe::byte_swap(security_info->page_descriptors[i].value);
    const uint32_t start = xex->base_address() + page * page_size;
    const uint32_t size = desc.page_count * page_size;
    page += desc.page_count;
    if (desc.info != XEX_SECTION_CODE) {
      continue;
    }
    const uint32_t pattern_bytes =
        static_cast<uint32_t>(signature.words.size() * 4);
    if (size < pattern_bytes) {
      continue;
    }
    for (uint32_t address = start; address + pattern_bytes <= start + size;
         address += 4) {
      if (MatchAt(memory, address, signature)) {
        if (++*out_match_count == 1) {
          found = address;
        }
      }
    }
  }
  return *out_match_count == 1 ? found : 0;
}


// Writes the module image and a sidecar describing it, for offline analysis
// (Ghidra) when adding signatures for a NUI library version.
void DumpModuleImage(Memory* memory, UserModule* module,
                     const std::vector<StaticLibrary>& libraries,
                     const std::filesystem::path& dir) {
  auto* xex = module->xex_module();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const uint32_t low = xex->base_address();
  const uint32_t high = low + xex->image_size();
  if (!low || high <= low) {
    XELOGW("NUI HLE: cannot dump {}: unknown image range", module->name());
    return;
  }
  const std::string base = StripXexExtension(module->name());
  const auto bin_path = dir / fmt::format("{}_{:08X}.bin", base, low);
  const auto json_path = dir / fmt::format("{}_{:08X}.json", base, low);
  FILE* f = xe::filesystem::OpenFile(bin_path, "wb");
  if (!f) {
    XELOGW("NUI HLE: cannot open {} for writing", xe::path_to_utf8(bin_path));
    return;
  }
  fwrite(memory->TranslateVirtual(low), 1, high - low, f);
  fclose(f);
  std::string json = "{\n";
  json += fmt::format(
      "  \"module\": \"{}\",\n  \"base\": \"0x{:08X}\",\n  \"end\": "
      "\"0x{:08X}\",\n",
      module->name(), low, high);
  json += "  \"static_libraries\": [";
  for (size_t i = 0; i < libraries.size(); ++i) {
    json += fmt::format("{}{{\"name\": \"{}\", \"version\": \"{}\"}}",
                        i ? ", " : "", libraries[i].name, libraries[i].version);
  }
  json += "],\n  \"code_ranges\": [";
  const auto* security_info = xex->xex_security_info();
  auto heap = memory->LookupHeap(xex->base_address());
  const uint32_t page_size = heap ? heap->page_size() : 0x1000;
  uint32_t page = 0;
  bool first = true;
  for (uint32_t i = 0; i < security_info->page_descriptor_count; ++i) {
    xex2_page_descriptor desc;
    desc.value = xe::byte_swap(security_info->page_descriptors[i].value);
    const uint32_t start = xex->base_address() + page * page_size;
    const uint32_t size = desc.page_count * page_size;
    page += desc.page_count;
    if (desc.info == XEX_SECTION_CODE) {
      json += fmt::format("{}{{\"start\": \"0x{:08X}\", \"size\": \"0x{:X}\"}}",
                          first ? "" : ", ", start, size);
      first = false;
    }
  }
  json += "]\n}\n";
  FILE* jf = xe::filesystem::OpenFile(json_path, "wb");
  if (jf) {
    fwrite(json.data(), 1, json.size(), jf);
    fclose(jf);
  }
  XELOGI("NUI HLE: dumped {} ({} bytes) to {}", module->name(), high - low,
         xe::path_to_utf8(bin_path));
}

const NuiLibrarySignatures* SelectSignatures(
    const std::string& module_name, const std::string& nui_version) {
  const std::string module_key = StripXexExtension(module_name);
  // Exact build first.
  for (const auto& lib : GetBuiltinNuiSignatures()) {
    if (lib.module_name && *lib.module_name &&
        StripXexExtension(lib.module_name) == module_key) {
      return &lib;
    }
  }
  for (const auto& lib : GetBuiltinNuiSignatures()) {
    if (lib.version && *lib.version && nui_version == lib.version) {
      return &lib;
    }
  }
  return nullptr;
}

}  // namespace

const NuiHleQuirks& GetActiveNuiHleQuirks() { return g_active_quirks; }

void AttachNuiHle(KernelState* kernel_state, UserModule* module) {
  if (!cvars::nui || !cvars::nui_hle) {
    return;
  }
  auto* nui_system = kernel_state->emulator()->nui_system();
  if (!nui_system || !module || !module->xex_module() ||
      !module->is_executable()) {
    return;
  }
  auto* xex = module->xex_module();
  auto* memory = kernel_state->memory();

  const auto libraries = ReadStaticLibraries(xex);
  std::string nui_version;
  std::string summary;
  for (const auto& lib : libraries) {
    if (lib.name == "NUI" || lib.name == "ST" || lib.name == "NUISP" ||
        lib.name == "NUIHNDL") {
      summary += fmt::format(" {}={}", lib.name, lib.version);
      if (lib.name == "NUI") {
        nui_version = lib.version;
      }
    }
  }
  if (nui_version.empty()) {
    XELOGD("NUI HLE: {} links no NUI library, nothing to do", module->name());
    return;
  }
  XELOGI("NUI HLE: {} links Kinect libraries:{}", module->name(), summary);
  if (!cvars::nui_dump_module_image.empty()) {
    DumpModuleImage(memory, module, libraries, cvars::nui_dump_module_image);
  }

  const NuiLibrarySignatures* signatures =
      SelectSignatures(module->name(), nui_version);
  if (!signatures) {
    XELOGW(
        "NUI HLE: no signatures for NUI library {} in {}; the title will see "
        "no Kinect. See docs/nui/adding_signatures.md",
        nui_version, module->name());
    nui_system->SetDevicePresent(false);
    return;
  }

  // Resolve every function first; an incompletely replaced runtime is worse
  // than none.
  struct Resolved {
    const NuiFunctionSignature* signature;
    uint32_t address;
    bool already_hooked;
  };
  std::vector<Resolved> resolved;
  std::vector<std::string> missing;
  for (const auto& signature : signatures->functions) {
    const std::string mode = signature.mode ? signature.mode : "hle";
    if (mode == "native") {
      continue;
    }
    uint32_t address = 0;
    bool already_hooked = false;
    if (signature.fixed_address) {
      if (xex->ContainsAddress(signature.fixed_address)) {
        const uint32_t first = xe::load_and_swap<uint32_t>(
            memory->TranslateVirtual(signature.fixed_address));
        if (first == kSyscallTrampolineWord) {
          address = signature.fixed_address;
          already_hooked = true;
        } else if (MatchAt(memory, signature.fixed_address, signature)) {
          address = signature.fixed_address;
        }
      }
    } else {
      uint32_t match_count = 0;
      address = FindByPattern(memory, xex, signature, &match_count);
      if (match_count > 1) {
        XELOGW("NUI HLE: signature for {} matched {} times, not hooking",
               signature.name, match_count);
      }
    }
    if (!address) {
      if (mode == "hle") {
        missing.push_back(signature.name);
      } else {
        XELOGD("NUI HLE: optional {} not found", signature.name);
      }
      continue;
    }
    resolved.push_back({&signature, address, already_hooked});
  }
  if (!missing.empty()) {
    std::string names;
    for (const auto& name : missing) {
      names += " " + name;
    }
    XELOGW(
        "NUI HLE: NUI {} in {} is known but these functions were not found:{}; "
        "the title will see no Kinect",
        nui_version, module->name(), names);
    nui_system->SetDevicePresent(false);
    return;
  }

  ResetNuiHleState(kernel_state);
  g_active_quirks = signatures->quirks;

  uint32_t hooked = 0;
  uint32_t skipped = 0;
  for (const auto& entry : resolved) {
    const std::string mode = entry.signature->mode ? entry.signature->mode
                                                   : "hle";
    if (entry.already_hooked) {
      XELOGI("NUI HLE: {} at {:08X} is already hooked (title hooks), skipping",
             entry.signature->name, entry.address);
      ++skipped;
      continue;
    }
    cpu::GuestFunction::ExternHandler handler =
        mode == "tripwire" ? LookupNuiTripwireHandler(entry.signature->name)
                           : LookupNuiHleHandler(entry.signature->name);
    if (!handler) {
      XELOGW("NUI HLE: no host implementation for {}, leaving it native",
             entry.signature->name);
      continue;
    }
    const std::string hook_name =
        std::string("NuiHLE::") + entry.signature->name;
    if (xex->InstallExternHook(entry.address, hook_name, handler, {})) {
      ++hooked;
    }
  }
  XELOGI("NUI HLE: NUI {} in {}: {} functions hooked, {} left to title hooks",
         nui_version, module->name(), hooked, skipped);
  nui_system->SetDevicePresent(true);
}

}  // namespace nui
}  // namespace kernel
}  // namespace xe
