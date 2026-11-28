// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nx2_resource_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/nx2_knobs.h"
#include "video_core/texture_cache/image_base.h"

namespace VideoCommon::ResourceTrace {
std::atomic<bool> enabled{};
std::atomic<bool> sampling{};

namespace {
std::mutex mutex;
Journal journal;
Journal critical_journal{2U << 20, 4096};
std::unordered_map<u64, std::array<u64, 3>> repeat_epochs;
u64 suppressed_repeats{};
std::map<std::pair<u64, u64>, u64> producer_epochs;
std::atomic<u64> identity{};
std::atomic<u64> frame{};
u64 sequence{};
u64 session{};
u32 draws{};
bool baseline_pending{};
bool baseline_saved{};
std::filesystem::path directory;
std::vector<std::pair<DAddr, u64>> buffers;

void Append(std::string_view kind, const std::string& fields, bool critical = false) {
    auto event =
        fmt::format(R"({{"seq":{},"frame":{},"kind":"{}"{}{}}})", ++sequence,
                    frame.load(std::memory_order_relaxed), kind, fields.empty() ? "" : ",", fields);
    if (critical)
        critical_journal.Push(event);
    journal.Push(std::move(event));
}

void Save(const char* name, const Journal* source = nullptr) {
    const auto& history = source ? *source : journal;
    std::ofstream out{directory / name, std::ios::trunc};
    out << fmt::format(
               R"({{"kind":"trace","version":1,"session":{},"frame":{},"dropped":{},"events":{},"bytes":{},"suppressed_repeats":{}}})",
               session, frame.load(), history.Dropped(), history.Size(), history.Bytes(),
               suppressed_repeats)
        << '\n';
    history.Write(out);
    out.close();
    if (out.fail()) {
        LOG_ERROR(Render_Vulkan, "Terrain trace: cannot write {}", (directory / name).string());
    } else {
        LOG_INFO(Render_Vulkan, "Terrain trace: saved {} ({} events, {} dropped)",
                 (directory / name).string(), history.Size(), history.Dropped());
    }
}

void SaveHistory() {
    std::error_code error;
    const auto previous = directory / "previous-history.jsonl";
    std::filesystem::remove(previous, error);
    error.clear();
    if (std::filesystem::exists(directory / "history.jsonl", error)) {
        std::filesystem::rename(directory / "history.jsonl", previous, error);
    }
    Save("history.jsonl");
    Save("critical-events.jsonl", &critical_journal);
}
} // namespace

u64 NewIdentity() {
    return identity.fetch_add(1, std::memory_order_relaxed) + 1;
}

void TickFrame() {
    static const char* const env = std::getenv("EDEN_NX2_TERRAIN_TRACE");
    const bool requested =
        (env && env[0] != '\0') || Settings::GetDebugKnobAt(Nx2Knob::TerrainTrace);
    std::scoped_lock lock{mutex};
    if (!requested) {
        if (enabled.exchange(false)) {
            sampling = false;
            Append("disabled", "");
            SaveHistory();
        }
        return;
    }
    if (!enabled.load()) {
        directory = env && env[0] != '\0'
                        ? std::filesystem::path{env}
                        : Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "terrain-trace";
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) {
            LOG_ERROR(Render_Vulkan, "Terrain trace: cannot create {}: {}", directory.string(),
                      error.message());
            return;
        }
        journal = Journal{};
        critical_journal = Journal{2U << 20, 4096};
        repeat_epochs.clear();
        producer_epochs.clear();
        suppressed_repeats = 0;
        sequence = 0;
        frame = 0;
        session = std::chrono::system_clock::now().time_since_epoch().count();
        draws = 0;
        buffers.clear();
        baseline_saved = baseline_pending = false;
        enabled = true;
        Append("enabled", R"("path":"emulated-gpu","sample_every_frames":60,"draw_limit":128)");
        LOG_INFO(Render_Vulkan, "Terrain resource trace enabled in {}", directory.string());
    }
    if (baseline_pending && !baseline_saved) {
        Save("baseline.jsonl");
        baseline_saved = true;
        baseline_pending = false;
    }
    std::error_code error;
    const auto clean = directory / "clean.trigger";
    if (std::filesystem::exists(clean, error)) {
        Append("checkpoint", R"("state":"clean")");
        Save("clean.jsonl");
        std::filesystem::remove(clean, error);
    }
    const auto trigger = directory / "dump.trigger";
    if (std::filesystem::exists(trigger, error)) {
        Append("checkpoint", "");
        SaveHistory();
        std::filesystem::remove(trigger, error);
    }
    ++frame;
    draws = 0;
    sampling = (frame.load() % 60 == 1) || !baseline_saved;
}

void Stop() {
    std::scoped_lock lock{mutex};
    sampling = false;
    if (enabled.exchange(false)) {
        Append("stopped", "");
        SaveHistory();
    }
}

bool BeginDraw(std::span<const u64> native_keys) {
    if (!Sampling() || !HasTerrainProgram(native_keys)) {
        return false;
    }
    if (draws++ >= 128) {
        if (draws == 129)
            Note("draw_limit", R"("limit":128)");
        return false;
    }
    baseline_pending = true;
    return true;
}

void Note(std::string_view kind, std::string fields, bool critical) {
    if (!Enabled())
        return;
    std::scoped_lock lock{mutex};
    if (Enabled())
        Append(kind, fields, critical);
}

void NoteProducer(std::string_view kind, std::string fields, u64 object, u64 program) {
    if (!Enabled())
        return;
    std::scoped_lock lock{mutex};
    if (producer_epochs.size() >= 4096)
        producer_epochs.clear();
    auto& previous = producer_epochs[{object, program}];
    const auto epoch = frame.load(std::memory_order_relaxed) / 60 + 1;
    if (previous == epoch) {
        ++suppressed_repeats;
        return;
    }
    previous = epoch;
    Append(kind, fields, true);
}

void Image(std::string_view event, const ImageBase& image, u32 slot, u64 related) {
    if (!Enabled())
        return;
    const auto& info = image.info;
    if (image.trace_identity == 0 && !((info.size.width == 260 && info.size.height == 260) ||
                                       (info.size.width == 1365 && info.size.height == 1365)))
        return;
    if (!image.trace_identity)
        image.trace_identity = NewIdentity();
    const bool normal = info.format == VideoCore::Surface::PixelFormat::R8G8_UNORM;
    const bool critical =
        normal && (event == "register" || event == "delete" || event == "upload-begin" ||
                   event == "device-unmap" || event == "gpu-unmap" || event == "gc-download" ||
                   event == "download" || event == "download-complete" || event == "read-trap-arm" ||
                   (event == "cpu-write" && True(image.flags & ImageFlagBits::GpuModified) &&
                    False(image.flags & ImageFlagBits::CpuModified)) ||
                   (event == "gpu-write" && image.first_written_frame == 0));
    const int repeat = event == "gpu-write"      ? 0
                       : event == "alias-source" ? 1
                       : event == "alias-copy"   ? 2
                                                 : -1;
    if (repeat >= 0) {
        std::scoped_lock lock{mutex};
        if (repeat_epochs.size() >= 4096)
            repeat_epochs.clear();
        auto& previous = repeat_epochs[image.trace_identity][repeat];
        const auto epoch = frame.load(std::memory_order_relaxed) / 60 + 1;
        if (previous == epoch) {
            ++suppressed_repeats;
            return;
        }
        previous = epoch;
    }
    Note(
        "image",
        fmt::format(
            R"("event":"{}","id":{},"slot":{},"related":{},"channel":{},"gpu":"{:#x}","device":"{:#x}","size":{},"format":{},"width":{},"height":{},"levels":{},"layers":{},"flags":{},"modified":{},"cpu_modified":{},"aliases":{},"overlaps":{})",
            event, image.trace_identity, slot, related, image.channel, image.gpu_addr,
            image.cpu_addr, image.guest_size_bytes, static_cast<u32>(info.format), info.size.width,
            info.size.height, info.resources.levels, info.resources.layers,
            static_cast<u32>(image.flags), image.modification_tick, image.cpu_modification_tick,
            image.aliased_images.size(), image.overlapping_images.size()),
        critical);
}

void WatchBuffer(DAddr address, u64 size) {
    std::scoped_lock lock{mutex};
    const auto range = std::pair{address, size};
    if (std::ranges::find(buffers, range) == buffers.end()) {
        if (buffers.size() == 256)
            buffers.erase(buffers.begin());
        buffers.push_back(range);
    }
}

bool WatchesBuffer(DAddr address, u64 size) {
    if (!Enabled())
        return false;
    std::scoped_lock lock{mutex};
    return std::ranges::any_of(buffers, [&](auto range) {
        return address < range.first + range.second && range.first < address + size;
    });
}
} // namespace VideoCommon::ResourceTrace
