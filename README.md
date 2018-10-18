# Tutorial on logining the docker instance
## Build your own docker instance
## Go to the GPU-management platform http://station.csgrandeur.com/gpu/faqs    
## Step1.Register using the inviting-code provided by @LiuNing
## Step2.Apply a port for accessing the servers.

# Tensorflow-Installation
## Step0. install CUDA Toolkit v8.0(It has already been install on our Servers,you can skip this step.)            
###  instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network))               
CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"       
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}           
dpkg -i ${CUDA_REPO_PKG}            
apt-get update           
apt-get -y install cuda           

## Step1. install cuDNN v6.0  
CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"   
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}   
tar -xzvf ${CUDNN_TAR_FILE}   
cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include    
cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/   
chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*   

## Step2. set environment variables   
### add the 2 paths below to ~/.bashrc:   
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}    
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}    

## Step3. download and install anaconda   
### select an compatible version of anaconda from "https://repo.continuum.io/archive/" or just use command below to download anaconda3:     
wget https://repo.continuum.io/archive/Anaconda3-2.4.0-Linux-x86_64.sh    

## Step4. install anaconda    
bash Anaconda3-2.4.0-Linux-x86_64.sh    

**Notice:   
Approve the licence at last:     
Do you approve the license terms? [yes|no]    
[no] >>> yes**       

### add the path below to ~/.bashrc   
$ export PATH=/root/anaconda3/bin:$PATH   



## Step5. download and install tensorflow-gpu   
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/    
conda config --set show_channel_urls yes    
  
conda create -n tensorflow-gpu python=3.6   
source activate tensorflow-gpu    
  

**Notice:the command below may fail,just try more times.**        
conda install tensorflow-gpu    

## Step6. update the pip source to accelerate the download speed
source activate tensorflow-gpu    
mkdir .pip    
vim .pip/pip.conf   
copy the following source to .pip/pip.conf             
(Notice: Enter the insert mode(press key "i") before you copy the following source to ./pip/pip.conf):     
"  
[global]    
index-url = http://mirrors.aliyun.com/pypi/simple/    
[install]   
trusted-host = mirrors.aliyun.com   
"  
After you copy this source to the pip.conf,press "esc" to escape from the insert mode,and then type into ":wq" to save ./pip/pip.conf
apt-get update      
### then you can install the libs you need using pip install xxx(e.g. pip install opencv-python)    
Problems you may encounter:   
After you pip install opencv-python,you may still fail to import cv2:   
$python   
>>>import cv2    
Traceback (most recent call last):  
  File "<stdin>", line 1, in <module>   
  File "/root/anaconda3/envs/tensorflow-gpu/lib/python3.5/site-packages/cv2/__init__.py", line 3, in <module>   
    from .cv2 import *    
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory   
  
### Solve the problem above using command as follow:    
apt-get update   
apt-get -y upgrade     
pip install opencv-python    
apt-get install libgtk2.0-dev -y   

## PS: use command "nvidia-smi" to monitor the usage of GPUs to see which processes that occupy the GPUs:  
fuser -v /dev/nvidia*  
this command may fail:   
$fuser command not found  
### install the psmisc which contains command fuser:  
apt-get install psmisc  

## Show tensorboard on your local machine
reference link[https://blog.csdn.net/bryant_meng/article/details/79153531]
# For Muti-GPUs synchronization(e.g. Batch_Normalization Synchronization)     
## Requirement:      
### 1.cuda9 + cudnn-v7.1 + tensorflow-1.10+ + nccl:         
#### commands for install cudnn-v7.1:     
wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz               
cp cuda/include/cudnn.h /usr/local/cuda/include             
cp cuda/lib64/libcudnn* /usr/local/cuda/lib64                     
chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 
#### commands for uninstall nccl
apt remove libnccl2 libnccl-dev
#### commands for install nccl
dpkg -i nccl-repo-ubuntu1604-2.3.5-ga-cuda9.0_1-1_amd64.deb              
apt update              
apt install libnccl2=2.3.5-2+cuda9.0 libnccl-dev=2.3.5-2+cuda9.0             
#### commands to check whehter nccl is correctly installed:
apt list | grep nccl
#### Using commands above,you may encounter error:
/sbin/ldconfig.real: /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7 is not a symbolic link
#### Solve the problem above use the method below:
1. check the link:                
sudo ldconfig -v            

2. Find the error:                  
/sbin/ldconfig.real: /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7 is not a symbolic link                   
libcudnn.so.7 -> libcudnn.so.7.0.5                 

3. Create the new link manually:               
sudo ln -sf /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7.0.5 /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7        
## Steps for this Muti-GPUs' environment configuration is similar with the configuration we mentioned before.
