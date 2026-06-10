"""
Usage in any notebook cell:
    %run ../../setup_env.py
"""
import importlib
import subprocess
import sys

REQUIRED = {
    "faiss":         "faiss-gpu-cu12",
    "datasets":      "datasets",
    "evaluate":      "evaluate",
    "accelerate":    "accelerate",
    "sentencepiece": "sentencepiece",
}

print("Checking environment...")
missing = []
for module, package in REQUIRED.items():
    try:
        importlib.import_module(module)
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ✗ {module}")
        missing.append(package)

if missing:
    print(f"\nInstalling {len(missing)} packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        *missing, "-q"
    ])
    print("Installation complete")
    print("IMPORTANT: restart the kernel now, then rerun this cell")
else:
    print("\nAll packages present — ready to go")

# apply faiss patch after install
try:
    import faiss
    import transformers.utils.import_utils as import_utils
    import transformers.utils as tu

    if hasattr(import_utils.is_faiss_available, "cache_clear"):
        import_utils.is_faiss_available.cache_clear()
    import_utils._faiss_available = True
    import_utils.is_faiss_available = lambda: True
    tu.is_faiss_available = lambda: True
    print(f"FAISS patch applied — version {faiss.__version__}")
except Exception as e:
    print(f"FAISS patch failed: {e}")