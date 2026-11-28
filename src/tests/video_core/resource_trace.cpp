// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <catch2/catch_test_macros.hpp>

#include "video_core/nx2_resource_trace.h"
#include "video_core/texture_cache/image_base.h"

using VideoCommon::ResourceTrace::Journal;

TEST_CASE("Resource history retains the newest complete events in order", "[resource_trace]") {
    Journal journal{12, 3};
    journal.Push("old");
    journal.Push("middle");
    journal.Push("new");
    REQUIRE(journal.Bytes() == 12);
    journal.Push("last");
    std::ostringstream out;
    journal.Write(out);
    REQUIRE(out.str() == "new\nlast\n");
    REQUIRE(journal.Dropped() == 2);
    REQUIRE(journal.Bytes() == 7);
}

TEST_CASE("Resource history bounds small events as well as large ones", "[resource_trace]") {
    Journal journal{1024, 2};
    journal.Push("a");
    journal.Push("b");
    journal.Push("c");
    std::ostringstream out;
    journal.Write(out);
    REQUIRE(out.str() == "b\nc\n");
    REQUIRE(journal.Size() == 2);
    REQUIRE(journal.Dropped() == 1);
}

TEST_CASE("Oversized resource events cannot erase retained history", "[resource_trace]") {
    Journal journal{4, 4};
    journal.Push("good");
    journal.Push("oversized");
    std::ostringstream out;
    journal.Write(out);
    REQUIRE(out.str() == "good\n");
    REQUIRE(journal.Dropped() == 1);
    REQUIRE(journal.Bytes() == 4);

    Journal disabled{8, 0};
    disabled.Push("a");
    REQUIRE(disabled.Size() == 0);
    REQUIRE(disabled.Dropped() == 1);
}

TEST_CASE("Resource recorder saves baseline and acknowledged checkpoints", "[resource_trace_io]") {
    namespace Trace = VideoCommon::ResourceTrace;
    const char* const output = std::getenv("EDEN_NX2_TERRAIN_TRACE");
    if (!output || !*output) {
        SKIP("Run with EDEN_NX2_TERRAIN_TRACE set to a disposable test directory");
    }
    const std::filesystem::path directory{output};
    Trace::TickFrame();
    REQUIRE(Trace::Enabled());
    REQUIRE(Trace::Sampling());
    const std::array<u64, 1> unrelated{123};
    const std::array<u64, 1> terrain{Trace::TERRAIN_PROGRAM_KEY};
    REQUIRE_FALSE(Trace::BeginDraw(unrelated));
    REQUIRE(Trace::BeginDraw(terrain));
    Trace::Note("draw", R"("draw":1,"stages":[])");

    VideoCommon::ImageBase first{VideoCommon::NullImageParams{}};
    first.info.size = {260, 260, 1};
    first.info.format = VideoCore::Surface::PixelFormat::R8G8_UNORM;
    Trace::Image("register", first, 9);
    REQUIRE(first.trace_identity != 0);
    Trace::Image("delete", first, 9);
    VideoCommon::ImageBase replacement{VideoCommon::NullImageParams{}};
    replacement.info.size = {260, 260, 1};
    replacement.info.format = VideoCore::Surface::PixelFormat::R8G8_UNORM;
    Trace::Image("register", replacement, 9);
    REQUIRE(replacement.trace_identity != first.trace_identity);

    Trace::WatchBuffer(0x1000, 0x100);
    REQUIRE(Trace::WatchesBuffer(0x0fff, 2));
    REQUIRE(Trace::WatchesBuffer(0x10ff, 1));
    REQUIRE_FALSE(Trace::WatchesBuffer(0x1100, 1));
    REQUIRE_FALSE(Trace::WatchesBuffer(0x0fff, 1));
    for (unsigned i = 1; i < 128; ++i)
        REQUIRE(Trace::BeginDraw(terrain));
    REQUIRE_FALSE(Trace::BeginDraw(terrain));
    Trace::TickFrame(); // Pin the first terrain frame before ordinary traffic overflows.
    // Compute writes can leave first_written_frame at zero. They must still coalesce.
    for (unsigned i = 0; i < 100; ++i)
        Trace::Image("gpu-write", first, 9);
    Trace::Note("normal-producer", R"("outcome":"submitted")", true);
    for (unsigned i = 0; i < 10; ++i)
        Trace::NoteProducer("normal-compute-producer", R"("width":64,"height":36)", 1, 100);
    Trace::NoteProducer("normal-compute-producer", R"("width":64,"height":36)", 1, 101);
    for (unsigned i = 0; i < 17000; ++i)
        Trace::Note("ordinary", R"("value":1)");
    std::ofstream{directory / "clean.trigger"}.close();
    std::ofstream{directory / "dump.trigger"}.close();
    Trace::TickFrame();
    REQUIRE(std::filesystem::exists(directory / "baseline.jsonl"));
    REQUIRE(std::filesystem::exists(directory / "clean.jsonl"));
    REQUIRE(std::filesystem::exists(directory / "history.jsonl"));
    REQUIRE_FALSE(std::filesystem::exists(directory / "clean.trigger"));
    REQUIRE_FALSE(std::filesystem::exists(directory / "dump.trigger"));
    Trace::Stop();
    REQUIRE_FALSE(Trace::Enabled());
    REQUIRE_FALSE(Trace::Sampling());
    REQUIRE(std::filesystem::exists(directory / "previous-history.jsonl"));
    std::ifstream critical{directory / "critical-events.jsonl"};
    REQUIRE(critical.good());
    std::string header;
    std::getline(critical, header);
    REQUIRE(header.find("\"dropped\":0") != std::string::npos);
    REQUIRE(header.find("\"suppressed_repeats\":108") != std::string::npos);
    unsigned lines = 0;
    std::string line;
    while (std::getline(critical, line))
        ++lines;
    REQUIRE(lines == 7); // Lifecycle events, first GPU write and distinct producers survive.
}
