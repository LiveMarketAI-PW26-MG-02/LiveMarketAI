#!/usr/bin/env python3
"""
=============================================================
  MULTIMODAL SECURITY SYSTEM — MASTER ORCHESTRATOR
  Coordinates Python → R → C++ → Java security layers
  All metadata exchanged via XML (no JSON)
=============================================================
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

BASE = Path(__file__).parent
SHARED = BASE / "shared"
SHARED.mkdir(exist_ok=True)


def banner(title: str):
    print()
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def run_step(label: str, cmd: list, cwd: Path = None):
    print(f"\n▶ {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd or BASE),
        capture_output=False, text=True
    )
    if result.returncode != 0:
        print(f"  ⚠ Exit code {result.returncode}")
    return result.returncode == 0


def main():
    banner("MULTIMODAL SECURITY SYSTEM — MASTER ORCHESTRATOR")
    print("  Layers:  Python → R → C++ → Java")
    print("  Format:  XML (no JSON)")
    print("  Security: AES-GCM + Feistel + PBKDF2 + SHA-256 + HMAC")

    # ── LAYER 1: Python ──────────────────────────────────────
    banner("LAYER 1 — PYTHON: ZIP + Stream Encryption + SHA-256")
    py_script = BASE / "python" / "secure_zip_creator.py"
    run_step("Python layer", [sys.executable, str(py_script)])

    # ── LAYER 2: R (if available) ────────────────────────────
    banner("LAYER 2 — R: Statistical Integrity Analysis")
    r_exe = shutil.which("Rscript")
    if r_exe:
        r_script = BASE / "r" / "integrity_verifier.R"
        run_step("R layer", [r_exe, str(r_script)])
    else:
        print("  ⚠ Rscript not found — skipping R layer (install R to enable)")

    # ── LAYER 3: C++ (compile + run) ─────────────────────────
    banner("LAYER 3 — C++: Feistel Cipher + CRC-32 + Binary Manifest")
    cpp_src = BASE / "cpp" / "secure_layer.cpp"
    cpp_bin = BASE / "cpp" / "mmsec_cpp"
    compiled = run_step(
        "Compile C++",
        ["g++", "-std=c++17", "-O2", "-o", str(cpp_bin), str(cpp_src)]
    )
    if compiled:
        enc_in  = str(SHARED / "archive.zip")
        enc_out = str(SHARED / "archive_cpp.mmsec")
        xml_out = str(SHARED / "manifest_cpp.xml")
        if (SHARED / "archive.zip").exists():
            run_step("C++ encrypt",  [str(cpp_bin), "encrypt",  enc_in, enc_out, "CppSecLayer#3!"])
            run_step("C++ analyse",  [str(cpp_bin), "analyse",  enc_out])
            run_step("C++ manifest", [str(cpp_bin), "manifest", enc_out, xml_out])
        else:
            run_step("C++ self-test", [str(cpp_bin)])
    else:
        print("  ⚠ C++ compilation failed — ensure g++ is installed")

    # ── LAYER 4: Java ────────────────────────────────────────
    banner("LAYER 4 — JAVA: AES-256-GCM + PBKDF2 + XML Manifest")
    java_src = BASE / "java" / "SecureZipManager.java"
    java_dir = BASE / "java"
    javac = shutil.which("javac")
    java  = shutil.which("java")
    if javac and java:
        compiled_j = run_step(
            "Compile Java",
            [javac, str(java_src)], cwd=java_dir
        )
        if compiled_j:
            run_step(
                "Run Java",
                [java, "-cp", str(java_dir), "SecureZipManager", str(SHARED)]
            )
    else:
        print("  ⚠ Java (javac/java) not found — skipping Java layer")

    # ── SUMMARY ──────────────────────────────────────────────
    banner("FINAL SUMMARY")
    outputs = list(SHARED.iterdir())
    xml_files = [f for f in outputs if f.suffix == ".xml"]
    sec_files = [f for f in outputs if f.suffix in (".mmsec", ".zip")]

    print(f"\n  Shared artifacts in: {SHARED}")
    print(f"\n  Security blobs (.mmsec):")
    for f in sec_files:
        sz = f.stat().st_size if f.exists() else 0
        print(f"    {f.name:<35} {sz:>10,} bytes")

    print(f"\n  XML manifests:")
    for f in xml_files:
        print(f"    {f.name}")

    print("\n  Security layers applied:")
    print("    [1] Python  — Stream cipher (CTR-HMAC-SHA256) + PBKDF2")
    print("    [2] R       — Statistical entropy / chi-sq / autocorrelation analysis")
    print("    [3] C++     — Feistel cipher (CBC-16r) + CRC-32 + SHA-256 binary check")
    print("    [4] Java    — AES-256-GCM + PBKDF2 (310K iter) + HMAC-SHA256 signing")
    print("\n  Data exchange format: XML (zero JSON)")
    print()


if __name__ == "__main__":
    main()
