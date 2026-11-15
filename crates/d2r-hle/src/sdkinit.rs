//! `nninitInitializeSdkModule` 调的服务层初始化：真机上各自连 Horizon 系统服务（IPC），
//! 宿主没有这些服务进程，因此原生接管为空操作（与参考工程 `hle_sdkinit.c` 一一对应）。
//!
//! 不接管的：`nn::os::InitializeVirtualAddressMemory` 属于 SDK 的 os 层；
//! `fs::SetLocalApplicationAccessLog` / `SetResultHandledByApplication` 只写 SDK 内部开关。

use d2r_abi::Res;
use d2r_macros::force;

#[force("nndiagStartup")]
extern "C" fn nndiag_startup() -> Res {
    d2r_diag::info!(d2r_diag::cat::DIAG, "init 接管：nndiagStartup（空操作）");
    Res::OK
}

#[force("_ZN2nn2oe10InitializeEv")]
extern "C" fn oe_initialize() -> Res {
    d2r_diag::info!(d2r_diag::cat::DIAG, "init 接管：oe::Initialize（空操作）");
    Res::OK
}

#[force("_ZN2nn3aoc10InitializeEv")]
extern "C" fn aoc_initialize() -> Res {
    d2r_diag::info!(d2r_diag::cat::DIAG, "init 接管：aoc::Initialize（空操作）");
    Res::OK
}

#[force("_ZN2nn3err26InitializeContextWriterApiEv")]
extern "C" fn err_initialize_context_writer_api() -> Res {
    d2r_diag::info!(
        d2r_diag::cat::DIAG,
        "init 接管：err::InitializeContextWriterApi（空操作）"
    );
    Res::OK
}

#[force("_ZN2nn2fs24InitializeForApplicationEv")]
extern "C" fn fs_initialize_for_application() -> Res {
    d2r_diag::info!(d2r_diag::cat::DIAG, "init 接管：fs::InitializeForApplication（空操作）");
    Res::OK
}

#[force("_ZN2nn4pctl10InitializeEv")]
extern "C" fn pctl_initialize() -> Res {
    d2r_diag::info!(d2r_diag::cat::DIAG, "init 接管：pctl::Initialize（空操作）");
    Res::OK
}

#[force("_ZN2nn4diag34InitializeApplicationAbortObserverEv")]
extern "C" fn diag_initialize_application_abort_observer() -> Res {
    d2r_diag::info!(
        d2r_diag::cat::DIAG,
        "init 接管：diag::InitializeApplicationAbortObserver（空操作）"
    );
    Res::OK
}
