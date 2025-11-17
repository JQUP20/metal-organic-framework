# Linux & Python 环境搭建指南

## 1. 前置准备

### 1.1 更新系统
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get upgrade -y

# CentOS/RHEL
sudo yum update -y
```

### 1.2 安装基础工具
```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential git wget curl vim

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git wget curl vim
```

## 2. 安装 Anaconda/Mamba

### 2.1 下载并安装 Miniforge (推荐)
Miniforge 默认使用 conda-forge 并内置 Mamba，速度更快。

```bash
# 下载 Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# 安装
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

# 初始化
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc
```

### 2.2 或安装传统 Anaconda
```bash
# 下载 Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh

# 安装
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p $HOME/anaconda3

# 初始化
$HOME/anaconda3/bin/conda init bash
source ~/.bashrc
```

## 3. 创建课程虚拟环境

### 3.1 使用 environment.yml (推荐)
```bash
# 克隆课程仓库
git clone <repository-url>
cd metal-organic-framework

# 创建环境
mamba env create -f environment.yml
# 或使用 conda
conda env create -f environment.yml

# 激活环境
conda activate ai-mof-course
```

### 3.2 使用 requirements.txt
```bash
# 创建新环境
conda create -n ai-mof-course python=3.10 -y
conda activate ai-mof-course

# 安装依赖
pip install -r requirements.txt
```

## 4. 安装外部 MOF 工具

### 4.1 安装 Zeo++
```bash
# 下载 Zeo++
cd ~
wget http://www.zeoplusplus.org/Zeo++-0.3.tar.gz
tar -xzf Zeo++-0.3.tar.gz
cd Zeo++-0.3

# 编译
make

# 添加到环境变量
echo 'export PATH=$HOME/Zeo++-0.3:$PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
network -h
```

### 4.2 安装 Poreblazer
```bash
# 下载 Poreblazer
cd ~
git clone https://github.com/SarkisovTeam/Poreblazer.git
cd Poreblazer

# 编译 (需要 gfortran)
sudo apt-get install gfortran  # Ubuntu/Debian
# 或
sudo yum install gcc-gfortran  # CentOS/RHEL

make

# 添加到环境变量
echo 'export PATH=$HOME/Poreblazer:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4.3 配置 MOFid
MOFid 已通过 pip 安装（mofid-cygwin），但需要额外配置：

```bash
# 验证安装
python -c "import mofid; print(mofid.__version__)"
```

## 5. 安装可视化工具

### 5.1 Avogadro (可选)
```bash
# Ubuntu/Debian
sudo apt-get install avogadro

# 或下载 AppImage
wget https://github.com/OpenChemistry/avogadrolibs/releases/download/1.95.1/Avogadro2-x86_64.AppImage
chmod +x Avogadro2-x86_64.AppImage
./Avogadro2-x86_64.AppImage
```

### 5.2 VESTA (可选)
```bash
cd ~
wget https://jp-minerals.org/vesta/archives/3.5.8/VESTA-gtk3.tar.bz2
tar -xjf VESTA-gtk3.tar.bz2
cd VESTA-gtk3
./VESTA
```

## 6. 验证安装

### 6.1 Python 环境验证
```bash
conda activate ai-mof-course
python << EOF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ase
import pymatgen
import sklearn
import torch
print("所有核心库安装成功！")
EOF
```

### 6.2 MOF 工具验证
```bash
# 检查 Zeo++
which network

# 检查 Python 包
python -c "import ase; from ase.io import read, write; print('ASE 工作正常')"
python -c "import pymatgen; print('Pymatgen 工作正常')"
```

## 7. Jupyter Notebook 配置

### 7.1 启动 JupyterLab
```bash
conda activate ai-mof-course
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 7.2 启用扩展
```bash
# 启用 nglview
jupyter nbextension enable --py --sys-prefix nglview
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

## 8. 常见问题

### Q1: conda 速度慢怎么办？
```bash
# 配置国内镜像源（清华源）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### Q2: pip 速度慢怎么办？
```bash
# 使用清华 PyPI 镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: GPU 支持问题
```bash
# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"

# 如果不可用，重新安装 PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 9. 下一步

环境搭建完成后，可以开始：
1. 浏览 `docs/theory/` 中的理论文档
2. 运行 `notebooks/` 中的示例 Notebook
3. 尝试 `src/` 中的代码示例

祝学习愉快！
