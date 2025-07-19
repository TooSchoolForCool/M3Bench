## Original Code
https://github.com/robotgradient/grasp_diffusion
## Installation
由于`theseus-ai`版本升级，源代码的安装方式存在问题，请参考以下方式安装

Clone repository
```python
git clone --recursive https://gitlab.mybigai.ac.cn/yansixu/grasp-diffusion.git
```
Create a conda environment
```python
conda create -n se3dif_env python==3.8
conda activate se3dif_env
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
nvcc --version && export CUDA_HOME=/usr/local/cuda
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```
Install mesh2sdf
```python
git clone https://github.com/robotgradient/mesh_to_sdf.git
cd mesh_to_sdf/
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```
NOTE: 您无需自己下载 pre-trained model，代码在运行过程中会自动下载
## Grasping Pose Sample
```python
cd grasp_diffusion/example/
python sample_grasping_poses.py 
```
NOTE: 请修改您的场景路径在 grasp_diffusion/example/scene.py 中