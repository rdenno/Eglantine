// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>

#include "core/core.h"
#include "core/hle/kernel/k_shared_memory.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/service.h"

namespace Service {

// User service returned by stpl:u 0 (nn::streamplay::detail::GetUserService). Right after opening
// it the SDK calls command 300 for the streaming HID shared memory: a copy handle plus a
// HidSharedMemoryConfig (u64 size, u64 offset). HidApplicationResourceHolder attaches and maps it
// read-only, then reads three input rings from base + offset + 0x6c000 + n * 0x6000. Zeroed memory
// means no remote players; a missing size aborts the boot (SMBW, 0xCA01). Everything else falls
// through to the nx2 auto stub (zero sizes, nothing streaming).
class INx2StreamPlayUserService final : public ServiceFramework<INx2StreamPlayUserService> {
public:
    explicit INx2StreamPlayUserService(Core::System& system_)
        : ServiceFramework{system_, "nx2:stpl:u::0"} {
        static const FunctionInfo functions[] = {
            {300, &INx2StreamPlayUserService::GetHidSharedMemory, "GetHidSharedMemory (guessed)"},
        };
        RegisterHandlers(functions);
    }

    ~INx2StreamPlayUserService() override {
        if (hid_memory != nullptr) {
            hid_memory->Close(system.Kernel());
        }
    }

private:
    static constexpr u64 HidSharedMemorySize = 0x80000;

    void GetHidSharedMemory(HLERequestContext& ctx) {
        LOG_WARNING(Service, "(STUBBED) stpl:u::0 {}, returning {:#x} bytes of HID shared memory",
                    ctx.GetCommand(), HidSharedMemorySize);
        if (hid_memory == nullptr) {
            auto& kernel_ref = system.Kernel();
            auto* const memory = Kernel::KSharedMemory::Create(kernel_ref);
            if (const Result result = memory->Initialize(
                    kernel_ref, system.DeviceMemory(), nullptr, Kernel::Svc::MemoryPermission::None,
                    Kernel::Svc::MemoryPermission::Read, HidSharedMemorySize);
                result.IsError()) {
                LOG_ERROR(Service, "Failed to make the stpl:u HID shared memory: {:#x}", result.raw);
                memory->Close(kernel_ref);
                IPC::ResponseBuilder rb{ctx, 2};
                rb.Push(result);
                return;
            }
            Kernel::KSharedMemory::Register(kernel_ref, memory);
            std::memset(memory->GetPointer(), 0, HidSharedMemorySize);
            hid_memory = memory;
        }
        IPC::ResponseBuilder rb{ctx, 6, 1};
        rb.Push(ResultSuccess);
        rb.Push<u64>(HidSharedMemorySize);
        rb.Push<u64>(0);
        rb.PushCopyObjects(ctx, *hid_memory);
    }

    Kernel::KSharedMemory* hid_memory{};
};

// Switch 2 stream play (GameShare), stpl:u. Command 0 (u32 0x12) opens the user service the SDK
// keeps for the process; without an object the first nn::streamplay call jumps through null.
class INx2StreamPlayService final : public ServiceFramework<INx2StreamPlayService> {
public:
    explicit INx2StreamPlayService(Core::System& system_)
        : ServiceFramework{system_, "nx2:stpl:u"} {
        static const FunctionInfo functions[] = {
            {0, &INx2StreamPlayService::GetUserService, "GetUserService (guessed)"},
        };
        RegisterHandlers(functions);
    }

private:
    void GetUserService(HLERequestContext& ctx) {
        LOG_WARNING(Service, "(STUBBED) stpl:u 0 [8]={:#x}, returning a user service",
                    ctx.CommandBuffer()[8]);
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<INx2StreamPlayUserService>(ctx, system);
    }
};

} // namespace Service
