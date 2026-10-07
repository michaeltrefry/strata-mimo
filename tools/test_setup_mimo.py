"""Tests for setup.py's MiMo-V2.6-Flash-RL choice (--family mimo, docs/MIMO.md): offered or refused by RAM, VRAM and
the GPU's vendor, the three pinned files with their sizes and SHA-256, files reused from Hugging Face's cache or
another folder instead of downloaded, the run config it writes (strata-mimo, its arguments, the tokenizer, the model
name), the start and update paths for that config, --dry-run, and the Qwen choices left as they were.  Mocked - no
GPU, no downloads, nothing written outside a temp folder.

    python tools/test_setup_mimo.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REV = "a5d1269fdcf9c3346934411f700038e9cd9d5102"
REPO = "ggml-org/MiMo-V2.6-Flash-RL-GGUF"
S1, S2, MTP = list(setup.MIMO_FILES)
PRO6000 = {"index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "vram_gb": 97887 / 1024,
           "arch": "120", "driver": "596.36"}


def quiet(fn, *args, **kw):
    """-> (exit code or return value, printed text)."""
    out = io.StringIO()
    code = None
    with contextlib.redirect_stdout(out):
        try:
            code = fn(*args, **kw)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class Pins(unittest.TestCase):
    def test_the_three_files(self):
        self.assertEqual(setup.MIMO_FILES, {
            "MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf":
                (5948640, "a34f63df32086bba7a14aa8a2f6c2f111f6e626ad7b10be18957e42d14d84738"),
            "MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf":
                (167363516512, "10b74ff85577de18b336718c19905a50bcbccfee9cae7b9edd446667084f623c"),
            "mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf":
                (2383728864, "cbcd8a22e62754b4133dc6f9ba87356268aa2d6377ee3e78be5d61172643969e"),
        })
        total = sum(b for b, _ in setup.MIMO_FILES.values())
        self.assertEqual(total, 169753194016)
        self.assertAlmostEqual(setup.MODELS["MXFP4"]["download_gb"], total / 1e9, places=1)

    def test_pinned_revision(self):
        fam = setup.FAMILIES["mimo"]
        self.assertEqual(setup.HF_REVISIONS[REPO], REV)
        self.assertEqual(fam["hf"], f"https://huggingface.co/{REPO}/resolve/{REV}/")
        self.assertEqual(fam["file"].format(q="MXFP4", i=1), S1)
        self.assertEqual(fam["file"].format(q="MXFP4", i=2), S2)
        self.assertEqual(fam["name"], "mimo-v2.6-flash-rl")
        self.assertTrue((ROOT / "data" / fam["profile"]).is_file())

    def test_qwen_choices_unchanged(self):
        sizes = lambda f: [m for m in setup.MODELS if f in setup.MODELS[m].get("families", ("qwen", "swift"))]  # noqa
        self.assertEqual(sizes("qwen"), ["Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S"])
        self.assertEqual(sizes("swift"), ["IQ2_XS", "IQ3_XXS"])
        self.assertEqual(sizes("coder"), ["IQ1_M"])
        self.assertEqual(sizes("unsloth"), ["UD-Q4_K_XL"])
        self.assertEqual(sizes("mimo"), ["MXFP4"])
        self.assertEqual(list(setup.FAMILIES)[:4], ["qwen", "swift", "coder", "unsloth"])   # qwen stays the default
        self.assertEqual(min(d["ram_gb"] for d in setup.MODELS.values()), 32)              # the smallest model's RAM


class Gates(unittest.TestCase):
    def test_ram(self):
        self.assertIsNone(setup.mimo_problem(255.6, 95.6, False))        # the 256 GB PC it was measured on (512 GB)
        self.assertIsNone(setup.mimo_problem(191.7, 95.6, False))        # a 192 GB PC as ram_gb() reads it
        self.assertIn("needs 192 GB", setup.mimo_problem(127.8, 95.6, False))
        self.assertIn("~150 GB", setup.mimo_problem(127.8, 95.6, False))

    def test_vram(self):
        self.assertIsNone(setup.mimo_problem(255.6, 31.8, False))        # a 32 GB card: allowed, with a warning
        self.assertIn("32 GB or more", setup.mimo_problem(255.6, 24.0, False))
        self.assertEqual(setup.mimo_notes(95.6, "120"), [])
        n = setup.mimo_notes(31.8, "120")
        self.assertEqual(len(n), 1)
        self.assertIn("speed is unmeasured", n[0])
        self.assertIn("untested", setup.mimo_notes(95.6, "89")[0])

    def test_amd(self):
        self.assertIn("NVIDIA-only", setup.mimo_problem(255.6, 95.6, True))

    def test_context(self):
        self.assertEqual(quiet(setup.mimo_context, None, 95.6, True)[0], 262144)
        self.assertEqual(quiet(setup.mimo_context, None, 47.8, True)[0], 65536)
        self.assertEqual(quiet(setup.mimo_context, 131072, 47.8, True)[0], 131072)
        code, out = quiet(setup.mimo_context, 393216, 95.6, True)
        self.assertEqual(code, 1)
        self.assertIn("at most 262144", out)


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def snap(self, hub: Path, rev=REV) -> Path:
        d = hub / "models--ggml-org--MiMo-V2.6-Flash-RL-GGUF" / "snapshots" / rev
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_hf_hub_dirs(self):
        self.assertEqual(setup.hf_hub_dirs({"HF_HOME": "E:/hf", "XDG_CACHE_HOME": "/c"}),
                         [Path("E:/hf/hub"), Path("/c/huggingface/hub")])
        self.assertEqual(setup.hf_hub_dirs({"HF_HUB_CACHE": "/x"})[0], Path("/x"))

    def test_hf_cached_pinned_revision_only(self):
        env = {"HF_HOME": str(self.t / "hf")}
        (self.snap(self.t / "hf" / "hub", "0" * 40) / S1).write_bytes(b"x")   # another revision: not used
        self.assertIsNone(setup.hf_cached(REPO, S1, env))
        (self.snap(self.t / "hf" / "hub") / S1).write_bytes(b"x")
        self.assertEqual(setup.hf_cached(REPO, S1, env), self.snap(self.t / "hf" / "hub") / S1)

    def test_locate(self):
        sizes = {n: (10 + i, "0" * 64) for i, n in enumerate(setup.MIMO_FILES)}
        models = self.t / "models" / "mimo-MXFP4"
        models.mkdir(parents=True)
        snap = self.snap(self.t / "hf" / "hub")
        (snap / S1).write_bytes(b"a" * 10)            # complete
        (snap / S2).write_bytes(b"a" * 5)             # short: downloaded instead
        (models / MTP).write_bytes(b"a" * 12)         # already here
        (snap / MTP).write_bytes(b"a" * 12)
        with mock.patch.dict(setup.MIMO_FILES, sizes):
            found = setup.mimo_locate(models, [], "mimo-MXFP4", env={"HF_HOME": str(self.t / "hf")})
        self.assertEqual(found, {S1: snap / S1, S2: None, MTP: models / MTP})

    def test_adopt_links_on_the_same_drive(self):
        src = self.t / "cache" / S1
        src.parent.mkdir()
        src.write_bytes(b"abc")
        dst = self.t / "models" / S1
        dst.parent.mkdir()
        p, out = quiet(setup.mimo_adopt, src, dst)
        self.assertEqual(p, dst)
        self.assertEqual(dst.read_bytes(), b"abc")
        self.assertTrue(setup.done(dst))
        with mock.patch.object(setup.os, "link", side_effect=OSError("another drive")):
            p, out = quiet(setup.mimo_adopt, src, self.t / "models" / S2)
        self.assertEqual(p, src)                      # used where it is, never copied
        self.assertIn("used where it is", out)
        self.assertFalse((self.t / "models" / S2).exists())

    def test_verify_once(self):
        f = self.t / S1
        f.write_bytes(b"strata" * 100)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        rec = self.t / "models" / "sha256-checked.json"
        ok, _ = quiet(setup.mimo_verify, f, 600, sha, rec)
        self.assertTrue(ok)
        self.assertIn(str(f), json.loads(rec.read_text()))
        with mock.patch("hashlib.sha256", side_effect=AssertionError("hashed again")):
            ok, _ = quiet(setup.mimo_verify, f, 600, sha, rec)
        self.assertTrue(ok)
        ok, out = quiet(setup.mimo_verify, f, 600, "0" * 64, rec)
        self.assertFalse(ok)
        self.assertIn("wrong SHA-256", out)
        self.assertTrue(f.exists())                   # a file it does not own is never deleted by the check
        ok, out = quiet(setup.mimo_verify, f, 601, sha, rec)
        self.assertFalse(ok)


class Main(unittest.TestCase):
    """setup.main() with --family mimo, every outside effect mocked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mimo-coding-answers.prof").parent.mkdir(parents=True)
        (self.t / "data" / "mimo-coding-answers.prof").write_bytes(b"STRP")
        self.downloads, self.runs, self.verified, self.built = [], [], [], []
        self.sizes = {n: (100 + i, hashlib.sha256(b"x" * (100 + i)).hexdigest())
                      for i, n in enumerate(setup.MIMO_FILES)}

    def tearDown(self):
        self.tmp.cleanup()

    def hf_snapshot(self, names=None) -> Path:
        snap = self.t / "hf" / "hub" / "models--ggml-org--MiMo-V2.6-Flash-RL-GGUF" / "snapshots" / REV
        snap.mkdir(parents=True, exist_ok=True)
        for n in names or setup.MIMO_FILES:
            (snap / n).write_bytes(b"x" * self.sizes[n][0])
        return snap

    def main(self, argv, ram=255.6, found=(PRO6000,), amd=(), free=500.0, verify=True):
        t = self.t

        def fake_build(gpu, yes, llama, dry=False):
            self.built.append((gpu["arch"], dry))
            eng = t / "engine-mimo"
            if not dry:
                eng.mkdir(exist_ok=True)
                (eng / setup.MIMO_EXE).write_bytes(b"")
                (eng / "BUILD.json").write_text(json.dumps({"engine": "mimo", "cuda_dirs": ["<cuda>"]}))
            return eng

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"x" * self.sizes[dst.name][0])
            setup.mark(dst)

        def fake_run(cmd, *a, **k):
            self.runs.append([str(x) for x in cmd])

        def fake_verify(p, size, sha, record):
            self.verified.append((p, size, sha))
            return verify

        patches = [
            mock.patch.object(setup, "ROOT", t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (t / "data", [])),
            mock.patch.object(setup, "data_folder_dry", lambda d: (t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: [dict(g) for g in found]),
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: free),
            mock.patch.object(setup, "free_gb_dry", lambda p: free),
            mock.patch.object(setup, "pip_install", mock.Mock()),
            mock.patch.object(setup, "get_llama_cpp", lambda: t / "llama.cpp"),
            mock.patch.object(setup, "get_prebuilt", mock.Mock(side_effect=AssertionError("Qwen engine"))),
            mock.patch.object(setup, "build_engine", mock.Mock(side_effect=AssertionError("Qwen engine"))),
            mock.patch.object(setup, "build_mimo_engine", fake_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "mimo_verify", fake_verify),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "hf_hub_dirs", lambda env=None: [t / "hf" / "hub"]),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port: t / f"run-{tag.lower()}.bat"),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(setup.MIMO_FILES, self.sizes),
            mock.patch.object(sys, "argv", ["setup.py", "--family", "mimo", "--yes", "--no-start",
                                            "--models-dir", str(t / "models"), *argv]),
            mock.patch("builtins.input", mock.Mock(side_effect=AssertionError("asked"))),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            code, out = quiet(setup.main)
        cfg = t / "strata-mimo-mxfp4.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)

    def test_install_downloads_the_pinned_files(self):
        code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        base = f"https://huggingface.co/{REPO}/resolve/{REV}/"
        self.assertEqual(self.downloads, [base + n for n in setup.MIMO_FILES])
        self.assertEqual([(p.name, s, h) for p, s, h in self.verified],
                         [(n, *self.sizes[n]) for n in setup.MIMO_FILES])
        self.assertEqual(self.built, [("120", False)])
        models = self.t / "models" / "mimo-MXFP4"
        pack = self.t / "data" / "packs" / "mimo-mxfp4"
        self.assertEqual(cfg, {
            "exe": str(self.t / "engine-mimo" / setup.MIMO_EXE),
            "args": ["--native", str(models / S1), "--mtp", str(models / MTP),
                     "--expert-profile", str(self.t / "data" / "mimo-coding-answers.prof"), "--max-context", "262144"],
            "cwd": str(self.t), "tokenizer": str(pack / "tokenizer"), "model_name": "mimo-v2.6-flash-rl",
            "log": str(self.t / "strata-mimo-mxfp4.log"), "lib_dirs": ["<cuda>"], "port": 8080, "family": "mimo",
            "gpu": 0, "gpus_asked": True})
        self.assertEqual(self.runs, [[sys.executable, str(self.t / "tools" / "strata_tokenizer.py"), "--gguf",
                                      str(models / S1), "--out", str(pack)]])
        self.assertIn("run-mimo-mxfp4.bat", out)

    def test_reuses_the_hugging_face_cache(self):
        snap = self.hf_snapshot()
        code, out, cfg = self.main(["--context", "131072"])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [])
        models = self.t / "models" / "mimo-MXFP4"
        self.assertTrue((models / S2).is_file())      # hard-linked (the same drive)
        self.assertEqual(cfg["args"][:4], ["--native", str(models / S1), "--mtp", str(models / MTP)])
        self.assertEqual(cfg["args"][-1], "131072")
        self.assertTrue((snap / S2).is_file())        # the cache is left as it was

    def test_cache_on_another_drive_is_used_in_place(self):
        snap = self.hf_snapshot()
        with mock.patch.object(setup.os, "link", side_effect=OSError("another drive")):
            code, out, cfg = self.main([])
        self.assertEqual(code, 0, out)
        self.assertEqual(self.downloads, [])
        self.assertEqual(cfg["args"][:4], ["--native", str(snap / S1), "--mtp", str(snap / MTP)])
        self.assertFalse((self.t / "models" / "mimo-MXFP4" / S2).exists())

    def test_a_bad_cached_file_is_downloaded_instead(self):
        self.hf_snapshot([S1])
        with mock.patch.object(setup.os, "link", side_effect=OSError("another drive")):
            code, out, cfg = self.main([], verify=False)
        self.assertEqual(code, 1)                     # the fake check fails the download too
        self.assertIn("not the pinned file", out)
        self.assertEqual(self.downloads[0], f"https://huggingface.co/{REPO}/resolve/{REV}/{S1}")

    def test_refused_below_192_gb_of_ram(self):
        code, out, cfg = self.main([], ram=127.8)
        self.assertEqual(code, 1)
        self.assertIn("MiMo needs 192 GB", out)
        self.assertIsNone(cfg)
        self.assertEqual((self.downloads, self.built), ([], []))

    def test_refused_below_32_gb_of_vram(self):
        code, out, cfg = self.main([], found=({**PRO6000, "name": "NVIDIA GeForce RTX 4090", "vram_gb": 24.0,
                                               "arch": "89"},))
        self.assertEqual(code, 1)
        self.assertIn("32 GB or more", out)
        self.assertIsNone(cfg)

    def test_a_48_gb_card_with_a_warning(self):
        code, out, cfg = self.main([], found=({**PRO6000, "name": "NVIDIA RTX 6000 Ada", "vram_gb": 47.5,
                                               "arch": "89"},))
        self.assertEqual(code, 0, out)
        self.assertIn("speed is unmeasured", out)
        self.assertIn("untested", out)
        self.assertEqual(cfg["args"][-1], "65536")     # under 80 GB: 64K by default
        self.assertEqual(self.built, [("89", False)])

    def test_two_cards_one_gpu_no_question(self):
        second = {**PRO6000, "index": 1}
        code, out, cfg = self.main([], found=(PRO6000, second))
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["gpu"], 0)
        self.assertNotIn("layer_split", cfg)

    def test_other_settings_left_out(self):
        code, out, cfg = self.main(["--vision", "yes", "--kv", "q4_0", "--draft-vocab", "en"])
        self.assertEqual(code, 0, out)
        self.assertIn("--vision does not apply to MiMo", out)
        self.assertIn("--kv does not apply to MiMo", out)
        self.assertNotIn("vision", cfg)
        self.assertNotIn("--kv", cfg["args"])

    def test_wrong_size(self):
        code, out, _ = self.main(["--model", "IQ2_XS"])
        self.assertEqual(code, 1)
        self.assertIn("no IQ2_XS", out)

    def test_dry_run_changes_nothing(self):
        self.hf_snapshot()
        with mock.patch.object(setup, "data_folder", mock.Mock(side_effect=AssertionError("data folder moved"))):
            code, out, cfg = self.main(["--dry-run"])
        self.assertEqual(code, 0, out)
        self.assertIsNone(cfg)
        self.assertEqual((self.downloads, self.runs, self.verified), ([], [], []))
        self.assertEqual(self.built, [("120", True)])
        self.assertFalse((self.t / "models").exists())
        self.assertIn('"model_name": "mimo-v2.6-flash-rl"', out)
        self.assertIn("would extract the tokenizer", out)
        self.assertIn("Dry run: nothing was installed", out)

    def test_dry_run_is_mimo_only(self):
        with mock.patch.object(sys, "argv", ["setup.py", "--family", "qwen", "--dry-run"]), \
                contextlib.redirect_stderr(io.StringIO()):
            code, _ = quiet(setup.main)
        self.assertEqual(code, 2)

    def test_check_lists_mimo(self):
        code, out, _ = self.main(["--check"])
        self.assertEqual(code, 0, out)
        self.assertIn("MXFP4    needs ~192 GB RAM, 32+ GB VRAM: fits (--family mimo", out)
        code, out, _ = self.main(["--check"], ram=63.7)
        self.assertIn("MXFP4    needs ~192 GB RAM, 32+ GB VRAM: does not fit (RAM: 64 GB", out)


class Config(unittest.TestCase):
    """The config setup wrote: found, started and updated the way a Qwen one is, with MiMo's differences."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        eng = self.t / "engine-mimo"
        eng.mkdir()
        (eng / setup.MIMO_EXE).write_bytes(b"")
        (eng / "BUILD.json").write_text(json.dumps({"engine": "mimo", "src": "same", "archs": [120]}))
        self.gguf = self.t / S1
        self.gguf.write_bytes(b"")
        (self.t / MTP).write_bytes(b"")
        self.cfg = {"exe": str(eng / setup.MIMO_EXE),
                    "args": ["--native", str(self.gguf), "--mtp", str(self.t / MTP), "--expert-profile", "p.prof",
                             "--max-context", "262144"],
                    "cwd": str(self.t), "tokenizer": str(self.t / "tok"), "model_name": "mimo-v2.6-flash-rl",
                    "port": 8080, "family": "mimo", "gpu": 0, "gpus_asked": True}
        self.path = self.t / "strata-mimo-mxfp4.json"
        self.path.write_text(json.dumps(self.cfg))

    def tearDown(self):
        self.tmp.cleanup()

    def test_found_like_a_qwen_config(self):
        with mock.patch.object(setup, "ROOT", self.t):
            self.assertEqual(setup.installed_configs(), [self.path])
        ch = setup.choices_from_config(self.path)
        self.assertEqual((ch["family"], ch["model"], ch["context"]), ("mimo", "MXFP4", 262144))

    def test_start(self):
        call = mock.Mock(return_value=0)
        with mock.patch.object(setup, "ROOT", self.t), \
                mock.patch.object(setup, "gpus", lambda: [dict(PRO6000), {**PRO6000, "index": 1}]), \
                mock.patch.object(setup, "source_hash", lambda parts: "same"), \
                mock.patch.object(setup, "refresh_draft_vocab", mock.Mock(side_effect=AssertionError("draft"))), \
                mock.patch.object(setup, "offer_together", mock.Mock(side_effect=AssertionError("together"))), \
                mock.patch.object(setup, "ensure_engine_for", mock.Mock(side_effect=AssertionError("Qwen engine"))), \
                mock.patch.object(setup, "build_mimo_engine", mock.Mock(side_effect=AssertionError("rebuilt"))), \
                mock.patch.object(setup.subprocess, "call", call):
            code, out = quiet(setup.start, self.path, None)
        self.assertEqual(code, 0, out)
        cmd = call.call_args[0][0]
        self.assertEqual(cmd[1:7], [str(self.t / "serve" / "server.py"), "--engine", "strata", "--config",
                                    str(self.path), "--port"])
        self.assertNotIn("--gpu", cmd)                 # the config's card (CUDA_VISIBLE_DEVICES, set by the server)
        self.assertIn("about 150 GB of experts", out)
        self.assertIn("GPU: GPU 0", out)

    def test_start_on_two_cards_is_refused(self):
        with mock.patch.object(setup, "ROOT", self.t), mock.patch.object(setup, "gpus", lambda: [dict(PRO6000)]):
            code, out = quiet(setup.start, self.path, None, [0, 1])
        self.assertEqual(code, 1)
        self.assertIn("one GPU", out)

    def test_a_changed_source_is_compiled_again(self):
        build = mock.Mock()
        with mock.patch.object(setup, "source_hash", lambda parts: "new"), \
                mock.patch.object(setup, "gpus", lambda: [dict(PRO6000)]), \
                mock.patch.object(setup, "get_llama_cpp", lambda: self.t / "llama.cpp"), \
                mock.patch.object(setup, "build_mimo_engine", build):
            quiet(setup.update_mimo_engine, self.cfg)
        self.assertEqual(build.call_args[0][0]["arch"], "120")
        build.reset_mock()
        with mock.patch.object(setup, "source_hash", lambda parts: "same"), \
                mock.patch.object(setup, "build_mimo_engine", build):
            quiet(setup.update_mimo_engine, self.cfg)
        build.assert_not_called()

    def test_update(self):
        upd = mock.Mock()
        with mock.patch.object(setup, "ROOT", self.t), \
                mock.patch.object(setup, "pip_install", lambda *a, **k: None), \
                mock.patch.object(setup, "update_installed_engine", lambda *a: None), \
                mock.patch.object(setup, "update_mimo_engine", upd), \
                mock.patch.object(setup, "refresh_draft_vocab", mock.Mock(side_effect=AssertionError("draft"))):
            code, out = quiet(setup.update_install, [self.path], mock.Mock(build=False, prebuilt=""))
        self.assertEqual(code, 0, out)
        upd.assert_called_once()
        self.assertIn("mimo-v2.6-flash-rl: up to date", out)

    def test_no_tuning(self):
        code, out = quiet(setup.calibrate_config, self.path)
        self.assertFalse(code)
        self.assertIn("no tuning for MiMo", out)


if __name__ == "__main__":
    unittest.main()
