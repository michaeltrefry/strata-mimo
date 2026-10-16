"""Check native-expert option ordering and MiMo's first-pass CMake wiring.

Run: python tools/test_cmake_native_experts.py
Requires CMake (or CMAKE_COMMAND pointing to it) and Ninja, but no compiler, GPU or downloads.
This configures a compiler-free fixture with the real option and MiMo wiring block
in source order and an empty cache. Target mutation is recorded; the full engine
and external dependencies are not configured or built.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
NINJA = shutil.which("ninja")


def native_chunks(text):
    option = list(re.finditer(
        r'^option\(STRATA_NATIVE_EXPERTS\s+[^\n]+\)$', text, re.MULTILINE))
    if len(option) != 1:
        raise AssertionError("Expected exactly one native-experts option")
    blocks = list(re.finditer(
        r'^  if\([^\n]+\)\n(?:(?!^  (?:if|endif)\().*\n)*'
        r'^  endif\(\)', text, re.MULTILINE))
    blocks = [block for block in blocks
              if "target_sources(strata_mimo PRIVATE src/mimo/prefill_moe.cu)" in block[0]]
    if len(blocks) != 1:
        raise AssertionError("Expected exactly one MiMo prefill-MoE wiring block")
    return sorted([option[0], blocks[0]], key=lambda match: match.start())


class NativeExpertsCmakeTests(unittest.TestCase):
    def test_option_precedes_every_use(self):
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        chunks = native_chunks(text)
        executable = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
        first_use = re.search(r'\bSTRATA_NATIVE_EXPERTS\b', executable)
        self.assertIsNotNone(first_use)
        self.assertTrue(executable[:first_use.start()].rstrip().endswith("option("),
                        "Native-experts option must be resolved before any use")
        self.assertTrue(chunks[0][0].startswith("option("))

    @unittest.skipUnless(CMAKE and NINJA, "CMake and Ninja required for clean-configure regression")
    def test_fresh_cache_mimo_dependencies(self):
        text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        chunks = "\n".join(match[0] for match in native_chunks(text))
        for backend in ("cuda", "hip", "cpu"):
            for native in (None, "ON", "OFF"):
                with self.subTest(backend=backend, native=native):
                    self.check_wiring(chunks, backend, native)

    def check_wiring(self, chunks, backend, native):
        # No -D for the default case: pre-seeding ON would hide the original bug.
        settings = (f'set(STRATA_ENABLE_CUDA {"ON" if backend == "cuda" else "OFF"})\n'
                    f'set(STRATA_ENABLE_HIP {"ON" if backend == "hip" else "OFF"})\n')
        if native is not None:
            settings += f'set(STRATA_NATIVE_EXPERTS {native} CACHE BOOL "")\n'
        expected = backend == "cuda" and native != "OFF"
        script = ('cmake_minimum_required(VERSION 3.24)\n'
                  'project(native_experts_regression LANGUAGES NONE)\n') + settings + r'''
function(target_sources target)
  if(NOT target STREQUAL "strata_mimo")
    message(FATAL_ERROR "Unexpected source target: ${target}")
  endif()
  set_property(GLOBAL APPEND PROPERTY mimo_sources "${ARGN}")
endfunction()
function(target_link_libraries target)
  if(NOT target STREQUAL "strata_mimo")
    message(FATAL_ERROR "Unexpected link target: ${target}")
  endif()
  set_property(GLOBAL APPEND PROPERTY mimo_links "${ARGN}")
endfunction()
''' + chunks + r'''
get_property(sources GLOBAL PROPERTY mimo_sources)
get_property(links GLOBAL PROPERTY mimo_links)
''' + (r'''
if(NOT sources STREQUAL "PRIVATE;src/mimo/prefill_moe.cu")
  message(FATAL_ERROR "Missing MiMo prefill source on first pass: ${sources}")
endif()
if(NOT links STREQUAL "PUBLIC;strata_mmq")
  message(FATAL_ERROR "Missing MiMo MMQ dependency on first pass: ${links}")
endif()
''' if expected else r'''
if(sources OR links)
  message(FATAL_ERROR "CUDA native wiring leaked into disabled backend: ${sources};${links}")
endif()
''')
        # Each case has a new working directory and process: no warmed cache, fetches,
        # generated engine files, compiler checks, or changes to a running executable.
        with tempfile.TemporaryDirectory(prefix="strata-cmake-native-") as tmp:
            path = Path(tmp) / "CMakeLists.txt"
            path.write_text(script, encoding="utf-8")
            result = subprocess.run([CMAKE, "-G", "Ninja", "-S", tmp,
                                     "-B", str(Path(tmp) / "build"),
                                     "-DCMAKE_MAKE_PROGRAM:FILEPATH=" + NINJA],
                                    cwd=tmp, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
