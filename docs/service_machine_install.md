# DREAM Service-Machine Setup

This document covers setup on the service machine (DREAM service environment + AnyGrasp environment).

## Target environment

- Ubuntu 22.04
- CUDA 12.1 is required for both the `dream` and `anygrasp` environments.
- The AnyGrasp PyTorch stack in this guide uses `cu121` wheels, so keep CUDA compatibility at 12.1.
- GPU requirement: at least 24 GB VRAM on a single GPU, or two GPUs with at least 16 GB VRAM each.
- AnyGrasp typically uses around 10 GB VRAM. If AnyGrasp and DREAM run on the same GPU, 24 GB VRAM is recommended. Otherwise, use two GPUs and split workloads.

## 1. DREAM environment (`conda`)

```bash
sudo apt update
sudo apt install -y libasound2-dev portaudio19-dev

cd ~/path/to/your/code
git clone https://github.com/BJHYZJ/DREAM.git --recursive
cd DREAM

conda create -n dream python=3.10 -y
conda activate dream

pip install -e ./src/
pip install rerun-sdk==0.26.1
pip install numpy==1.23.5
pip install transforms3d==0.3.1
pip install zstd
```

Recommended (avoid user-site pollution in `dream` env):

```bash
conda activate dream
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate && conda activate dream
python -c "import site, sys; print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('\\n'.join(sys.path))"
```

## 2. AnyGrasp environment (`conda`)

From this section onward, all steps are for the AnyGrasp stack.

```bash
export PYTHONNOUSERSITE=1
conda create -n anygrasp python=3.10 -y
conda activate anygrasp

cd ~/DREAM_ws/DREAM/third_party/segment-anything-2
pip install -e .

pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 \
  -f https://download.pytorch.org/whl/cu121/torch_stable.html

pip install ipython scipy==1.10.1 scikit-learn==1.4.0 pandas==2.0.3 \
  hydra-core opencv-python openai-clip timm matplotlib==3.7.2 imageio \
  open3d numpy-quaternion more-itertools pyliblzfse einops transformers \
  pytorch-lightning wget gdown tqdm zmq torch_geometric numpy==1.23.0
```

### 2.1 MinkowskiEngine

Reference: https://github.com/Julie-tang00/Common-envs-issues/blob/main/Cuda12-MinkowskiEngine

Environment check:

```bash
python -c "import sys; import torch; print('Python version:', sys.version); print('Torch version:', torch.__version__); print('Torch CUDA version:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available());"
gcc --version | head -n 1 | cut -d' ' -f3
```

Install dependencies:

```bash
pip install --upgrade setuptools==59.8.0
sudo apt install -y build-essential python3-dev libopenblas-dev
pip install ninja
```

If your build fails with `__to_address` related errors on Ubuntu 22.04 / GCC 11,
patch `/usr/include/c++/11/bits/shared_ptr_base.h`:

- Replace:
  - `auto __raw = __to_address(__r.get())`
- With:
  - `auto __raw = std::__to_address(__r.get())`

Install MinkowskiEngine:

```bash
cd ~/DREAM_ws
git clone https://github.com/pccws/MinkowskiEngine
cd MinkowskiEngine
python setup.py install
```

If compilation fails, patch these headers in the cloned repo:

- `src/convolution_kernel.cuh`: `#include <thrust/execution_policy.h>`
- `src/coordinate_map_gpu.cu`: `#include <thrust/unique.h>` and `#include <thrust/remove.h>`
- `src/spmm.cu`: `#include <thrust/execution_policy.h>`, `#include <thrust/reduce.h>`, `#include <thrust/sort.h>`
- `src/3rdparty/concurrent_unordered_map.cuh`: `#include <thrust/execution_policy.h>`

### 2.2 Other dependencies

```bash
conda activate anygrasp

pip install --upgrade --no-deps --force-reinstall scikit-learn==1.4.0
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install numpy==1.23.0

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install git+https://github.com/graspnet/graspnetAPI.git
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
```

Set AnyGrasp binary modules (`python==3.10` example):

```bash
cd ~/DREAM_ws/DREAM
cp third_party/anygrasp_sdk/grasp_detection/gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so src/anygrasp_manipulation/gsnet.so
cp third_party/anygrasp_sdk/license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so src/anygrasp_manipulation/lib_cxx.so
```

Build pointnet2 and finalize Python deps:

```bash
cd ~/DREAM_ws/DREAM/src/anygrasp_manipulation/pointnet2
python -m pip install --upgrade "pip>=23" "setuptools>=64"
pip install --no-build-isolation -e .

pip install rerun-sdk==0.26.1
pip install numpy==1.23.5
pip install transforms3d==0.3.1
```

Prepare checkpoints directory:

```bash
cd ~/DREAM_ws/DREAM
mkdir -p checkpoints/anygrasp
mkdir -p checkpoints/sam2
```

### 2.3 AnyGrasp license

```bash
sudo apt install -y libssl-dev
cd ~/DREAM_ws/DREAM/src/anygrasp_manipulation
./anygrasp_license_registration/license_checker -f
```

If missing `libcrypto.so.1.1`:

```bash
wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb
sudo apt-get install -f
```

License instructions:

- [AnyGrasp license README](../src/anygrasp_manipulation/anygrasp_license_registration/README.md)
- Put license file under [`src/anygrasp_manipulation/license`](../src/anygrasp_manipulation/license)
- Put AnyGrasp model checkpoints under [`checkpoints/anygrasp`](../checkpoints/anygrasp)

Check license status:

```bash
cd ~/DREAM_ws/DREAM/src/anygrasp_manipulation
./anygrasp_license_registration/license_checker -c license/licenseCfg.json
```
