// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"

// A Switch 2 GPU starts on a command list some time after the game submits it, and games rely on
// that: Hyrule Warriors: Age of Imprisonment writes the vertices and indices of new UI labels about
// 0.5-2 ms after submitting the draws that use them (one frame of black tiles and missing text on
// the map otherwise). The GPU thread holds each list until it is this old, unless a game thread
// blocks on the GPU first (then it isn't writing). It only waits when it has caught up with the
// game; a list that has queued that long already runs at once.
namespace VideoCommon::Nx2GpuLatency {

/// Bumped whenever a game thread starts waiting for GPU progress.
inline std::atomic<u64> g_guest_waits{0};

inline void NoteGuestWait() {
    g_guest_waits.fetch_add(1, std::memory_order_release);
}

[[nodiscard]] inline u64 GuestWaits() {
    return g_guest_waits.load(std::memory_order_acquire);
}

/// EDEN_NX2_GPU_LATENCY_US=<n> sets the hold (0 turns it off).
[[nodiscard]] inline std::chrono::microseconds Latency() {
    static const std::chrono::microseconds latency{[] {
        const char* const value = std::getenv("EDEN_NX2_GPU_LATENCY_US");
        return value != nullptr ? std::strtol(value, nullptr, 10) : 3000L;
    }()};
    return latency;
}

} // namespace VideoCommon::Nx2GpuLatency
