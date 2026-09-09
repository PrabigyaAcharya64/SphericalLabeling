# SphericalLabeling (maintained fork)

Fork of [ilwoolyu/SphericalLabeling](https://github.com/ilwoolyu/SphericalLabeling),
updated to run on current Python, NumPy, SciPy, PyTorch and libigl.

The upstream repository targets Python 3.6-era dependencies and does not run as
published on a modern environment. This fork restores a missing file, ports a
dead dependency, and fixes several API changes. The method and all
hyperparameters are unchanged.

Reference: Lyu I, Bao S, Hao L, Yao J, Miller JA, Voorhies W, Taylor WD,
Bunge SA, Weiner KS, Landman BA. *Labeling lateral prefrontal sulci using
spherical data augmentation and context-aware training.* NeuroImage 229
(2021) 117758. https://doi.org/10.1016/j.neuroimage.2021.117758
---

## Requirements

- Python 3.12 or 3.13 (the `libigl` wheel is `cp312-abi3`; earlier versions will not resolve)
- Linux or macOS
- A CUDA GPU for training. Inference runs on CPU.

Mesh generation and the model forward pass have been verified on macOS ARM.
Training has not been run; see [Verification status](#verification-status).

## Setup

```bash
git clone https://github.com/PrabigyaAcharya64/SphericalLabeling.git
cd SphericalLabeling

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python gen_ico.py
```

`gen_ico.py` writes `mesh_files/icosphere_0..7` as both `.pkl` and `.vtk`,
16 files in total. The `.pkl` files hold the precomputed sparse convolution
operators used by the network; the `.vtk` files are the plain geometry used by
SphericalRemesh during data preparation.

Verify:

```python
import pickle
d = pickle.load(open("mesh_files/icosphere_5.pkl", "rb"))
print(d["V"].shape)      # (10242, 3)
```

Icosphere levels: ico4 = 2,562 vertices, ico5 = 10,242, ico6 = 40,962.

`mesh_files/` is gitignored. `icosphere_7.pkl` is 118 MB, above GitHub's
100 MB limit, and the files regenerate in seconds.

## Pipeline overview

This repository is the network only. Producing its input requires two external
tools, distributed together in the `ilwoolyu/cmorph` Docker and Singularity
images.

| Stage | Tool | Output |
|---|---|---|
| MRI to cortical surfaces | FreeSurfer | `lh.curv`, `lh.sulc`, `lh.sphere`, `lh.inflated.H` |
| Surface registration | HSD | `lh.coeff.txt` (spherical harmonic coefficients) |
| Resampling + augmentation | SphericalRemesh | `.dat` files on a fixed icosphere |
| Training and inference | this repo | model checkpoints, per-label probabilities |

Augmentation happens inside the resampling step. A single SphericalRemesh call
reconstructs the registration warp at each spherical harmonic degree from 0 to
`--deg`, producing one resampled copy per degree. Degree 0 is close to
unregistered, the maximum degree is fully registered, and the intermediates are
partial registrations. `aug0` is used for validation and testing.

**Labels must be resampled with `--nneighbor`.** Without it they are
interpolated barycentrically, silently averaging distinct label values into
meaningless ones. Features use `--ftype double`; labels use
`--ftype int16 --nneighbor`.

Expected input filename grammar:

```
<subject>.<hemi>.<augTag>.<featname>.dat
```

for example `subj_1.lh.aug0.curv.dat`.

## Changes from upstream

### Restored file

`meshcnn/mesh_utils.py` is imported by `meshcnn/mesh.py` but is absent from the
upstream repository, so `gen_ico.py` fails immediately on a clean clone. The
file is vendored here from
[maxjiang93/ugscnn](https://github.com/maxjiang93/ugscnn) (MIT), the project
`meshcnn/` is adapted from.

### Ported dependency

`meshcnn/mesh.py` and `meshcnn/mesh_utils.py` imported `pyigl`, an obsolete
libigl binding that no longer installs. The current `libigl` package provides
the same functions under the name `igl`, with two API differences: results are
returned rather than written into preallocated `igl.eigen` containers, and
numpy arrays are accepted directly, so the `p2e`/`e2p` conversion helpers are
no longer needed.

`construct_matrices` in `meshcnn/mesh.py` was rewritten accordingly. Five
calls are affected: `grad`, `cotmatrix`, `per_face_normals`, `doublearea`,
`massmatrix`. `per_face_normals` now takes a fallback normal for degenerate
faces, passed as `[0, 0, 0]`.

`p2e` and `e2p` remain in `mesh_utils.py` but are dead code; they reference
`igl.eigen`, which no longer exists.

### Import and control-flow fixes

- `meshcnn/mesh.py`: the relative import of `mesh_utils` is wrapped in a
  `try`/`except`, since `gen_ico.py` loads `mesh` as a top-level module via
  `sys.path.append` while the file was written for package context.
- `gen_ico.py`: VTK export sat inside the `except ImportError` branch, so it
  ran only when mesh generation failed. Moved out of the branch; it now runs on
  both paths.

### Compatibility fixes

| File | Was | Now |
|---|---|---|
| `loader.py` | `.astype(np.int)` | `.astype(int)` |
| `meshcnn/mesh_utils.py` | `isinstance(digits, np.float)` | `isinstance(digits, float)` |
| `meshcnn/mesh_utils.py` | `m.dtype.type == np.bool` | `m.dtype.type == bool` |
| `meshcnn/utils.py` | `sparse.coo.coo_matrix` | `sparse.coo_matrix` |
| `meshcnn/utils.py` | `torch.sparse.FloatTensor(...)` | `torch.sparse_coo_tensor(...)` |
| `meshcnn/ops.py` | `sparse2tensor(pkl['G'])` | `sparse2tensor(pkl['G'].tocoo())` |
| `train.py`, `test.py` | `torch.load(path)` | `torch.load(path, weights_only=False)` |

`np.int`, `np.float` and `np.bool` were removed in NumPy 2.0. The
`scipy.sparse.coo` namespace is deprecated. `torch.sparse.FloatTensor` is
superseded by `torch.sparse_coo_tensor`.

Two of these are worth explaining further.

The `ops.py` change follows from the libigl port: `igl.grad` now returns a
`csc_matrix` where the old binding returned `coo`, and `sparse2tensor` asserts
`coo`. `pkl['L']` and `pkl['F2V']` were already converted at their call sites,
so this matches the existing idiom.

The `torch.load` change is needed because PyTorch 2.6 flipped `weights_only`
to default true. These checkpoints are full pickles containing optimizer state,
which the restricted unpickler rejects.

### Deliberately not changed

- `train.py`: `from torch.autograd import Variable`. Unused, still imports.
- `train.py`: `scheduler.step(epoch)`. Emits a deprecation warning but works;
  changing it risks altering learning-rate schedule semantics.
- `meshcnn/utils.py`: `torch.LongTensor([m.row, m.col])` triggers a performance
  warning about constructing tensors from lists of arrays. Harmless at ico5;
  worth revisiting if sparse tensor construction becomes slow at ico6 or ico7.
- No algorithm, formula, hyperparameter or default value was modified.

## Verification status

**Verified on macOS ARM, Python 3.12:**

- `gen_ico.py` produces 16 files from a clean checkout; `icosphere_5.pkl` has
  `V` of shape `(10242, 3)`
- All 11 modules import cleanly, with `SyntaxWarning` promoted to error
- `SphericalUNet` forward pass on CPU: input `(2, 3, 10242)` returns
  `(2, 8, 10242)` at `max_level=5`, all values finite
- `loader.py` parses the filename grammar, performs k-fold selection, and
  collates batches of `(4, 1, 10242)` data with `(4, 10242)` labels, using
  synthetic `.dat` files

**Not verified:**

- End-to-end training and inference. `train.py` and `test.py` call `.cuda()`
  unconditionally and require processed data plus a GPU.
- Checkpoint loading. No `.pth` file was available, so the `weights_only=False`
  change is reasoned but unexercised.
- Data preparation (FreeSurfer, HSD, SphericalRemesh). Requires those tools and
  real subject data.

## Attribution

Upstream: [ilwoolyu/SphericalLabeling](https://github.com/ilwoolyu/SphericalLabeling)
by Ilwoo Lyu. `meshcnn/` is adapted from
[maxjiang93/ugscnn](https://github.com/maxjiang93/ugscnn) (MIT). Original
licence terms apply; see `LICENSE`.