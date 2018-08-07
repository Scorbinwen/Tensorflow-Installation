# Tensorflow-Installation

# install CUDA Toolkit v8.0
# instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network))
CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
apt-get update
apt-get -y install cuda

# install cuDNN v6.0
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}
tar -xzvf ${CUDNN_TAR_FILE}
cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include
cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/
chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

# set environment variables
add the following 2 paths to ~/.bashrc:
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# download and install anaconda
select an compatible version of anaconda from "https://repo.continuum.io/archive/"
or just use command below to download anaconda3:
wget https://repo.continuum.io/archive/Anaconda3-2.4.0-Linux-x86_64.sh
# install anaconda
dash Anaconda3-2.4.0-Linux-x86_64.sh
follow the prompt...

# download and install tensorflow-gpu
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes

conda create -n tensorflow-gpu python=3.6
source activate tensorflow-gpu

conda install tensorflow-gpu
