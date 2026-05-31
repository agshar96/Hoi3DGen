# Install

This project uses three separate Conda environments:

- `SANA_ENV`: text-to-3D setup, SANA, FLUX retexturing dependencies, Grounded-SAM-2, and Hunyuan3D-2 mesh generation.
- `HY21_ENV`: Hunyuan 2.1 painting/rendering setup. Also used for rendering, which is the first step of segmentation.
- `CAMHMR_ENV`: segmentation support and SMPL alignment through CameraHMR.

For the text-to-3D workflow, use `SANA_ENV` and `HY21_ENV`.

For segmentation and aligning SMPL, use `CAMHMR_ENV`.

## Prerequisites

Make sure Conda is installed and available in your shell before starting.

Run all commands from the repository root unless a step explicitly changes directories.

---

## 1. SANA_ENV

Create and activate the environment:

```bash
conda create -n SANA_ENV python=3.11.0 -y
conda activate SANA_ENV
```

Optionally install CUDA toolkit if you prefer not to use the built-in `nvcc`:

```bash
conda install -c nvidia cuda-toolkit=12.4 -y
```

Install `xformers`:

```bash
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
```

### Install SANA

```bash
cd external/Sana

python -m pip install "setuptools<81"
pip install -e . --no-build-isolation
pip install "diffusers==0.33.0"
pip install "transformers==4.49.0"

cd ../..
```

### Install FLUX retexturing dependency

```bash
pip install git+https://github.com/asomoza/image_gen_aux.git
```

### Install Grounded-SAM-2

```bash
cd external/Grounded-SAM-2

pip install -e .
conda install -c conda-forge gcc_linux-64=10 gxx_linux-64=10
pip install --no-build-isolation -e grounding_dino
pip install supervision
pip install pycocotools==2.0.8
python -m pip install --no-cache-dir "timm>=1.0.13"

cd ../..
```

### Install Hunyuan3D-2 mesh generation

```bash
cd external/HunYuan3D-2

pip install -r requirements.txt
pip install -e .
```

If you see an error related to `TORCH_CUDA_ARCH_LIST`, export it before building the custom rasterizer:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
```

Build the custom rasterizer:

```bash
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../
```

Return to the repository root if needed:

```bash
cd ../..
```

---

## 2. HY21_ENV

Create and activate the environment:

```bash
conda create -n HY21_ENV python=3.10.0 -y
conda activate HY21_ENV
```

Install Hunyuan 2.1 dependencies:

```bash
cd external/Hunyuan-2.1

pip install -r requirements.txt
python -m pip install --force-reinstall "setuptools<82"
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi
```

> `pip install -r requirements.txt` can take a long time.

### Optional: fix `libXfixes.so.3` error

Only run the following commands if you get a `libXfixes.so.3` error:

```bash
conda install -c conda-forge -y \
  xorg-libxfixes \
  xorg-libxi \
  xorg-libxrender \
  xorg-libxext \
  xorg-libsm \
  xorg-libice \
  libgl \
  libxkbcommon

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### Build Hunyuan 2.1 rasterizer and renderer

```bash
cd hy3dpaint/custom_rasterizer
pip install -e . --no-build-isolation
cd ../../

cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..
```

Download the Real-ESRGAN checkpoint:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

Install the rendering dependency used for the first segmentation step:

```bash
pip install easydict
```

Return to the repository root if needed:

```bash
cd ../..
```

---

## 3. CAMHMR_ENV

Create and activate the environment:

```bash
conda create -n CAMHMR_ENV python=3.10 -y
conda activate CAMHMR_ENV
```

Install PyTorch:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

Install setup tools compatibility package:

```bash
python -m pip install "setuptools<81"
```

Install CameraHMR:

```bash
cd external/CameraHMR

pip install -r requirements.txt --no-build-isolation
bash scripts/fetch_demo_data.sh
```

Install additional dependencies:

```bash
pip install open3d

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
```

Return to the repository root if needed:

```bash
cd ../..
```

---

## Environment usage summary

| Environment | Purpose |
|---|---|
| `SANA_ENV` | SANA, FLUX retexturing dependencies, Grounded-SAM-2, Hunyuan3D-2 mesh generation |
| `HY21_ENV` | Hunyuan 2.1 painting/rendering and first segmentation rendering step |
| `CAMHMR_ENV` | Segmentation support and SMPL alignment with CameraHMR |