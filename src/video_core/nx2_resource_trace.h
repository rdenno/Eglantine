// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <ostream>
#include <span>
#include <string>
#include <string_view>

#include "common/common_types.h"

namespace VideoCommon {
struct ImageBase;

namespace ResourceTrace {

// Metadata only. Large game resources remain in explicit frame captures.
class Journal {
public:
    explicit Journal(size_t byte_limit = 8U << 20, size_t event_limit = 16384)
        : byte_limit{byte_limit}, event_limit{event_limit} {}

    void Push(std::string event) {
        if (event.size() > byte_limit || event_limit == 0) {
            ++dropped;
            return;
        }
        while (!events.empty() &&
               (bytes + event.size() > byte_limit || events.size() >= event_limit)) {
            bytes -= events.front().size();
            events.pop_front();
            ++dropped;
        }
        bytes += event.size();
        events.push_back(std::move(event));
    }

    void Write(std::ostream& out) const {
        for (const auto& event : events) {
            out << event << '\n';
        }
    }

    [[nodiscard]] size_t Dropped() const {
        return dropped;
    }
    [[nodiscard]] size_t Bytes() const {
        return bytes;
    }
    [[nodiscard]] size_t Size() const {
        return events.size();
    }

private:
    size_t byte_limit;
    size_t event_limit;
    size_t bytes{};
    size_t dropped{};
    std::deque<std::string> events;
};

extern std::atomic<bool> enabled;
extern std::atomic<bool> sampling;
[[nodiscard]] inline bool Enabled() {
    return enabled.load(std::memory_order_relaxed);
}
[[nodiscard]] inline bool Sampling() {
    return sampling.load(std::memory_order_relaxed);
}

/// Native program key of the terrain program the trace follows.
inline constexpr u64 TERRAIN_PROGRAM_KEY = 0x44c5c4f3fdaaf250ULL;

/// Whether a draw's native program keys (one per stage, 0 where unbound) include the terrain
/// program.
[[nodiscard]] inline bool HasTerrainProgram(std::span<const u64> native_keys) {
    return std::ranges::find(native_keys, TERRAIN_PROGRAM_KEY) != native_keys.end();
}

/// Rasterizer thread only: enables with EDEN_NX2_TERRAIN_TRACE=<directory> or debug knob13.
void TickFrame();
void Stop();
[[nodiscard]] bool BeginDraw(std::span<const u64> native_keys);
[[nodiscard]] u64 NewIdentity();
/// JSON fields without outer braces. Callers supply only fixed field names and numeric values.
void Note(std::string_view kind, std::string fields, bool critical = false);
void NoteProducer(std::string_view kind, std::string fields, u64 object, u64 program);
void Image(std::string_view event, const ImageBase& image, u32 slot, u64 related = 0);
void WatchBuffer(DAddr address, u64 size);
[[nodiscard]] bool WatchesBuffer(DAddr address, u64 size);
} // namespace ResourceTrace
} // namespace VideoCommon
