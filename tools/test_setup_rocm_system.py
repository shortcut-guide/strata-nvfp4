"""Tests for setup.py's system ROCm search outside the /opt/rocm layout: a distribution install (Fedora in
/usr with lib64, Debian/Ubuntu in /usr with the multiarch folder) was not recognised at all - rocm_root looked
only at $ROCM_PATH or /opt/rocm and only in lib/ for libhipblas - and build_engine_hip then passed the
/opt-layout compiler path (<root>/llvm/bin/clang++) to CMake even for a root that keeps clang++ elsewhere, so
a PC with a working system ROCm 7 (the tested path for an RX 6800 XT, gfx1030) fell back to the unvalidated
gfx103X-all wheels or stopped on a compiler that does not exist.  No network, no GPU.

    python -m unittest tools.test_setup_rocm_system
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def make_rocm(base: Path, lib_sub: str, compiler_rel: str) -> Path:
    """A complete system ROCm 7 under `base`: hipcc, libhipblas in `lib_sub`, the HIP development files
    (hip-lang where CMake looks for it in that layout, hip_runtime.h) and the HIP compiler at `compiler_rel`."""
    for rel, text in (("bin/hipcc", ""),
                      (f"{lib_sub}/libhipblas.so.3", ""),
                      (f"{lib_sub}/cmake/hip-lang/hip-lang-config.cmake", ""),
                      ("include/hip/hip_runtime.h", ""),
                      ("include/rocm-core/rocm_version.h",
                       "#define ROCM_VERSION_MAJOR 7\n#define ROCM_VERSION_MINOR 2\n"),
                      (compiler_rel, "")):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return base


class SystemRocmLayouts(unittest.TestCase):
    def root_for(self, base: Path):
        """rocm_root with only `base` as a candidate: the wheels path must not be reached (it would try pip)."""
        with mock.patch.object(setup, "rocm_system_candidates", return_value=[base]), \
                mock.patch.object(setup, "run", side_effect=AssertionError("the wheels were used")):
            return setup.rocm_root(["gfx1030"])

    def test_opt_rocm_layout_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            base = make_rocm(Path(d), "lib", "llvm/bin/clang++")
            self.assertEqual(self.root_for(base), (base, [str(base / "lib")]))
            self.assertEqual(setup.hip_compiler(base), base / "llvm" / "bin" / "clang++")

    def test_fedora_usr_layout_lib64(self):
        with tempfile.TemporaryDirectory() as d:
            base = make_rocm(Path(d), "lib64", "lib64/llvm/bin/clang++")
            self.assertEqual(self.root_for(base), (base, [str(base / "lib64")]))
            self.assertEqual(setup.hip_compiler(base), base / "lib64" / "llvm" / "bin" / "clang++")

    def test_debian_usr_layout_multiarch_and_bin_compiler(self):
        with tempfile.TemporaryDirectory() as d:
            base = make_rocm(Path(d), "lib/x86_64-linux-gnu", "bin/clang++")
            self.assertEqual(self.root_for(base), (base, [str(base / "lib" / "x86_64-linux-gnu")]))
            self.assertEqual(setup.hip_compiler(base), base / "bin" / "clang++")

    def test_candidates_order(self):
        with mock.patch.dict(setup.os.environ, {"ROCM_PATH": "/somewhere/rocm"}):
            cands = setup.rocm_system_candidates()
        self.assertEqual(cands[0], Path("/somewhere/rocm"))
        self.assertEqual(cands[1], Path("/opt/rocm"))
        self.assertEqual(cands[-2:], [Path("/usr/local"), Path("/usr")])
        self.assertEqual(len(cands), len(set(cands)))          # no root is tried twice

    def test_versioned_dirs_sort_by_version_numbers_not_names(self):
        with tempfile.TemporaryDirectory() as d:
            opt = Path(d)
            for name in ("rocm-6.4", "rocm-10.0", "rocm-7.2", "rocm-7.10"):
                (opt / name).mkdir()
            (opt / "rocm-docs").mkdir()                # not a versioned install: not a candidate
            self.assertEqual([x.name for x in setup.rocm_versioned_dirs(opt)],
                             ["rocm-10.0", "rocm-7.10", "rocm-7.2", "rocm-6.4"])

    def test_no_compiler_is_none_not_the_opt_path(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "bin").mkdir()
            self.assertIsNone(setup.hip_compiler(base))

    def test_bitcode_dir_follows_the_layout(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "lib64" / "llvm" / "amdgcn" / "bitcode").mkdir(parents=True)
            self.assertEqual(setup.rocm_bitcode_dir(base), base / "lib64" / "llvm" / "amdgcn" / "bitcode")
            self.assertEqual(setup.rocm_bitcode_dir(Path(d) / "none"), Path(d) / "none" / "amdgcn" / "bitcode")

    def test_lib_dirs_need_a_shared_library(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "lib64" / "cmake").mkdir(parents=True)     # CMake packages only: not a library folder
            (base / "lib").mkdir()
            (base / "lib" / "libhipblas.so.3").write_text("")
            self.assertEqual(setup.rocm_lib_dirs(base), [base / "lib"])


if __name__ == "__main__":
    unittest.main()
