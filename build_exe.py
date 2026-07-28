#!/usr/bin/env python3
"""
build_exe.py - Standalone EXE Packaging Script using PyInstaller

Usage:
  python build_exe.py --target knifedge
  python build_exe.py --target opdviewer
  python build_exe.py --target hyperviewer
  python build_exe.py --target all
"""

import os
import sys
import argparse
import subprocess

TARGETS = {
    "knifedge": {
        "script": "knifedge.py",
        "name": "KnifeEdgeViewer",
        "hidden_imports": ["scipy.signal", "scipy.optimize", "scipy.interpolate", "skimage", "tifffile", "PyQt5"]
    },
    "opdviewer": {
        "script": "OPDviewer/OPDViewer.py",
        "name": "OPDViewer",
        "hidden_imports": ["scipy.signal", "scipy.optimize", "skimage", "tifffile", "PyQt5"]
    },
    "hyperviewer": {
        "script": "HyperViewer.py",
        "name": "HyperViewer",
        "hidden_imports": ["scipy.signal", "scipy.optimize", "skimage", "tifffile", "PyQt5"]
    },
    "unmixer": {
        "script": "unmix/Unmixer.py",
        "name": "Unmixer",
        "hidden_imports": ["scikit-learn", "pywavelets", "scipy", "PyQt5"]
    }
}

def build_executable(target_key, mode="folder"):
    if target_key not in TARGETS:
        print(f"Unknown target '{target_key}'. Valid targets: {list(TARGETS.keys())}")
        return

    cfg = TARGETS[target_key]
    script = cfg["script"]
    name = cfg["name"]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconsole",
        "--clean",
        "--name", name
    ]

    if mode == "onefile":
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    for imp in cfg["hidden_imports"]:
        cmd.extend(["--hidden-import", imp])

    cmd.append(script)

    print(f"\n=======================================================")
    print(f" Building Standalone EXE for '{name}' ({script})")
    print(f" Mode: {'Single File (.exe)' if mode == 'onefile' else 'Standalone Folder (dist/)'}")
    print(f"=======================================================\n")

    res = subprocess.run(cmd)
    if res.returncode == 0:
        dist_path = os.path.abspath(os.path.join("dist", name))
        print(f"\n✅ BUILD SUCCESSFUL!")
        print(f"Executable output directory: {dist_path}")
    else:
        print(f"\n❌ BUILD FAILED with exit code {res.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Standalone EXE with PyInstaller")
    parser.add_argument("--target", choices=list(TARGETS.keys()) + ["all"], default="knifedge", help="App target to build")
    parser.add_argument("--mode", choices=["folder", "onefile"], default="folder", help="folder (fast startup) or onefile (single EXE)")
    args = parser.parse_args()

    if args.target == "all":
        for t in TARGETS.keys():
            build_executable(t, mode=args.mode)
    else:
        build_executable(args.target, mode=args.mode)
