# MLSCHOOL_2026_TIFR
2026 edition of HEP_ML School in India

# Installation Guide — Machine Learning Tutorial

This guide walks you through setting up the Python environment for the tutorial, including **PyTorch**, **PyTorch Geometric (PyG)**, **PyTorch Lightning**, and **SciPy**.

> **Time required:** ~10–15 minutes (longer if downloading CUDA-enabled PyTorch).

> **Last updated:** May 2026 — covers Python 3.14, PyTorch 2.11, Lightning 2.6.

---

## 1. Prerequisites

| Requirement | Recommended | Notes |
|---|---|---|
| Operating System | Linux, macOS, or Windows 10/11 | WSL2 recommended on Windows for GPU work |
| Python | **3.12 or 3.13** (3.14 also works — see below) | Minimum supported by PyTorch is 3.10 |
| Disk space | ~5 GB free | More if installing CUDA toolkit |
| GPU (optional) | NVIDIA GPU with CUDA 11.8 or newer | Check with `nvidia-smi` |

### Which Python version?

The latest stable Python is **3.14.3** (released February 2026), but for ML work I recommend **Python 3.12 or 3.13** because:

- All ML libraries (PyTorch, PyG, Lightning, scikit-learn) have full wheels — no compiling from source.
- Python 3.14 is supported by PyTorch 2.11+, but CUDA/GPU wheels for `cp314` are still rolling out across the ecosystem; some PyG accelerated extensions may not be available yet.
- Python 3.10 and 3.11 also work fine — use them if your existing environment already has one.

If you want to use 3.14, go ahead — CPU installs will work, and most GPU installs do too. Just be ready to fall back to 3.13 if a specific package is missing wheels.

Verify your Python version:

```bash
python --version
```

If you don't have Python or it's older than 3.10, install it from [python.org](https://www.python.org/downloads/) or via your system package manager.

---

## 2. Create an Isolated Environment

Always install into a fresh virtual environment to avoid clashing with system packages.

### Option A — `venv` (built-in)

```bash
python -m venv ml-tutorial
# Activate it:
source ml-tutorial/bin/activate        # Linux / macOS
ml-tutorial\Scripts\activate           # Windows (cmd)
ml-tutorial\Scripts\Activate.ps1       # Windows (PowerShell)
```

### Option B — `conda` (recommended if you already use Anaconda/Miniconda)

```bash
conda create -n ml-tutorial python=3.12
conda activate ml-tutorial
```

### Option C — `uv` (fast, modern alternative)

If you have [uv](https://docs.astral.sh/uv/) installed, it's significantly faster:

```bash
uv venv ml-tutorial --python 3.12
source ml-tutorial/bin/activate        # Linux / macOS
ml-tutorial\Scripts\activate           # Windows
```

Once activated, your shell prompt should show `(ml-tutorial)`.

Upgrade pip before installing anything else:

```bash
pip install --upgrade pip setuptools wheel
```

---

## 3. Install PyTorch

**Install PyTorch first** — PyTorch Geometric and Lightning both depend on it, and the install command depends on which PyTorch/CUDA combo you pick.

Pick **one** of the commands below. These match PyTorch 2.11's available wheel channels.

### CPU-only (works everywhere; pick this if unsure)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### CUDA 12.8 (recommended for most modern NVIDIA GPUs)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### CUDA 13.0 (newest — for the latest NVIDIA drivers)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### CUDA 12.6 (still widely supported)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### CUDA 11.8 (older driver / older GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Apple Silicon (M1/M2/M3/M4):** the CPU command above already enables Metal Performance Shaders (MPS). No extra setup needed.
>
> **Always check the latest:** the PyTorch install matrix occasionally shifts which CUDA versions ship wheels — visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) if any of the above 404s.

Verify the install:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 4. Install SciPy and Core Scientific Stack

```bash
pip install numpy scipy scikit-learn matplotlib pandas jupyter
```

This pulls in everything used in the tutorial notebooks: array math, classical ML utilities, plotting, dataframes, and Jupyter.

Verify:

```bash
python -c "import scipy; print('SciPy', scipy.__version__)"
```

---

## 5. Install PyTorch Geometric

For PyG **2.3 and newer**, the basic install is a single line:

```bash
pip install torch-geometric
```

That's enough for most of the tutorial. The lines below are only needed if you hit a "requires `torch-scatter` / `torch-sparse`" message when running an example.

### Optional accelerated dependencies

These compiled extensions speed up sparse operations. The wheel index URL **must match your PyTorch and CUDA version**.

Look up your installed PyTorch version first:

```bash
python -c "import torch; print(torch.__version__)"
```

Then pick the matching index URL. Replace `${TORCH}` with your version (e.g. `2.11.0`) and `${CUDA}` with `cpu`, `cu118`, `cu126`, `cu128`, or `cu130`:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

**Concrete example** — PyTorch 2.11.0 on CPU:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.11.0+cpu.html
```

The full list of available wheels lives at [data.pyg.org/whl](https://data.pyg.org/whl/).

---

## 6. Install PyTorch Lightning

Lightning gives you a clean `Trainer` abstraction so you can avoid writing training loops by hand.

> **Naming note:** The package was renamed from `pytorch-lightning` to `lightning` a few years ago. Both packages still exist on PyPI, but **`lightning` is the modern recommended package**. New code should `import lightning as L`. Old `import pytorch_lightning as pl` code still works.

### Install

```bash
pip install lightning
```

### Apple Silicon (M1/M2/M3/M4) — extra step

Set these two environment variables before running `pip install`:

```bash
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1
pip install lightning
```

This avoids a known issue when one of Lightning's transitive dependencies (`grpcio`) builds from source on macOS.

### Optional extras

Lightning ships several extra dependency groups. Install only what you need:

```bash
# Just the core Trainer (recommended for the tutorial)
pip install lightning

# Add common extras (logging, profiling, etc.)
pip install "lightning[extra]"

# Add support for distributed strategies like DeepSpeed
pip install "lightning[strategies]"
```

### Verify

```bash
python -c "import lightning as L; print('Lightning', L.__version__)"
```

---

## 7. Verify the Full Install

Save the snippet below as `check_install.py` and run it with `python check_install.py`:

```python
import sys
print("Python:", sys.version.split()[0])

import numpy, scipy, sklearn
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

import torch_geometric
print("PyTorch Geometric:", torch_geometric.__version__)

import lightning as L
print("Lightning:", L.__version__)

# Tiny end-to-end smoke test: build a graph + a Lightning module
from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1.0], [0.0], [1.0]])
data = Data(x=x, edge_index=edge_index)
print("Sample graph:", data)

class TinyModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
    def forward(self, x):
        return self.layer(x)

module = TinyModule()
print("Sample Lightning module instantiated:", type(module).__name__)
print("\n✅ Environment ready.")
```

If every line prints without errors and you see `✅ Environment ready.`, you're good to start the tutorial.

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'torch'` when installing `torch-geometric` or `lightning`**
You skipped step 3. Install PyTorch first, then PyG and Lightning.

**PyG runtime error mentioning `torch-scatter` or `torch-sparse`**
Install the optional accelerated dependencies from step 5, making sure the `${TORCH}` and `${CUDA}` tags match your installed PyTorch exactly. A mismatch is by far the most common cause of PyG install pain.

**`torch.cuda.is_available()` returns `False` despite having a GPU**
Your driver may be older than the CUDA build you installed. Run `nvidia-smi` to see the supported CUDA version and reinstall PyTorch with a matching `cu118`/`cu126`/`cu128`/`cu130` index URL.

**No CUDA wheel found for Python 3.14**
The PyTorch ecosystem is still rolling out `cp314` GPU wheels. If `pip install` only gives you a CPU build, either wait for the wheel to ship or downgrade to Python 3.13 in your virtual environment:

```bash
deactivate
rm -rf ml-tutorial
python3.13 -m venv ml-tutorial
source ml-tutorial/bin/activate
```

**`grpcio` build errors on macOS during Lightning install**
You're missing the env vars in step 6. Set `GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1` and `GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1` and reinstall.

**SSL or proxy errors on `pip install`**
If you're behind a corporate firewall, set `HTTPS_PROXY` or pass `--proxy http://your.proxy:port` to pip.

**Slow installs on Windows**
The PyG compiled extensions don't always have Windows wheels. Either use WSL2 or stick to the CPU-only install without `torch-scatter`/`torch-sparse` — the pure-Python fallbacks in PyG 2.3+ work fine for the tutorial.

**Want to start over?**
Just delete the environment and recreate it:

```bash
deactivate
rm -rf ml-tutorial          # or: conda env remove -n ml-tutorial
```

---

## 9. Quick Reference — All Commands

CPU-only setup with Python 3.12, copy-pasteable:

```bash
python3.12 -m venv ml-tutorial
source ml-tutorial/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn matplotlib pandas jupyter
pip install torch-geometric
pip install lightning
```

GPU setup (CUDA 12.8) with Python 3.12:

```bash
python3.12 -m venv ml-tutorial
source ml-tutorial/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-learn matplotlib pandas jupyter
pip install torch-geometric
pip install lightning
```

That's it — launch `jupyter notebook` and open the first tutorial.

