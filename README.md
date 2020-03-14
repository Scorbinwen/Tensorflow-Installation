# Tutorial on virtual envs configuration
DeepLearning framework Tensorflow/Pytorch installation tutorial for newbees...
* [Prerequisites](#prerequisites)  
* [Tensorflow Installation](#Tensorflow-Installation)  
* [Muti-GPUs synchronization](#Muti-GPUs-synchronization)  
* [Server Port Table](#Server-Port-Table)  
* [Others](#Others)
# Prerequisites
## Build your own docker instance
* Go to the GPU-management platform http://station.csgrandeur.com/gpu/faqs    
* Register an account using the inviting-code provided by @LiuNing
* Select an available server and build a new docker instance.

# DL-Framework configuration
## Step0. Choose compatible CUDA cudnn version.
## Step0. CUDA Installation

* CUDA:  
```
The NVIDIA® CUDA® Toolkit provides a development environment for creating high performance GPU-accelerated applications. With the CUDA Toolkit, you can develop, optimize and deploy your applications on GPU-accelerated embedded systems, desktop workstations, enterprise data centers, cloud-based platforms and HPC supercomputers. The toolkit includes GPU-accelerated libraries, debugging and optimization tools, a C/C++ compiler and a runtime library to deploy your application.
```
* cuDNN:  
```
The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
```
* CUDA & cuDNN compatible table:   
Refer to [https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html]

* install CUDA Toolkit[https://developer.nvidia.com/cuda-toolkit] v8.0,or any other version you need(It has already been install on our Servers,you can skip this step.)

###  instructions from https://developer.nvidia.com/cuda-downloads (linux -> x86_64 -> Ubuntu -> 16.04 -> deb (network))
#### Fetch the .deb cuda package.
```
$CUDA_REPO_PKG="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"       
$wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG}           
```
#### Install cuda 
```
$dpkg -i ${CUDA_REPO_PKG}            
$apt-get update           
$apt-get -y install cuda          
$vim ~/.bashrc
Append the following two env paths:
```
$export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}    
$export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}  
[Explanation]: ${PATH:+:${PATH}} means that if PATH exists and PATH is not null then append the directory ${PATH} to PATH.
```
```
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
## Step1. install cuDNN v6.0
Install the compatible(with CUDA) cuDNN version:
```
$CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"   
$wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}   
$tar -xzvf ${CUDNN_TAR_FILE}   
$cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include    
$cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/   
$chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*   
```
## Step2. set environment variables   
## Step3. download and install anaconda   
### download anaconda3: 
```
$wget https://repo.continuum.io/archive/Anaconda3-2.4.0-Linux-x86_64.sh    
```
[**Recommendation**]:You can  update conda to the lastest version using command as follow:
```
conda update conda
```
## Step4. install anaconda    
```
$bash Anaconda3-2.4.0-Linux-x86_64.sh    
```
**Notice:   
Approve the licence at last and follow the installation navigation:     
Do you approve the license terms? [yes|no]    
[no] >>> yes**       

## Step5. Create virtual envs
### Add tsinghua conda source to accelerate the download speed.
```
$conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/    
$conda config --set show_channel_urls yes    

$conda create -n YOUR_ENV_NAME python=3.6   
$source activate YOUR_ENV_NAME    
```

## Step6. update the pip source to accelerate the download speed
```
$source activate YOUR_ENV_NAME
$mkdir .pip    
$vim .pip/pip.conf   
```
copy the following source to .pip/pip.conf             

(Notice: Enter the insert mode(press key "i") before you copy the following source to ./pip/pip.conf):     
"  
[global]    
index-url = http://mirrors.aliyun.com/pypi/simple/    
[install]   
trusted-host = mirrors.aliyun.com   
"  
After you copy this source to the pip.conf,press "esc" to escape from the insert mode(and enter into normal mode),
and then type  ":wq" to save ./pip/pip.conf and exit                        
     
### then you can install the libs you need using pip install xxx(e.g. pip install opencv-python)    

 # Server Port Table
 |   Server IP   |    Port    |    Owner   |   Public  |
 |---------------|------------|------------|-----------|
 |192.168.50.113 |    33010   |    None    |Yes
 |192.168.50.113 |    33020   |  Wenxiaobin|No
 |192.168.50.113 |    33030   |  Zouzhongquan|No
 |192.168.50.113 |    33040   | RenHui     |Yes
 |192.168.50.113 |    33050   |Not Allocated|No
 |192.168.50.113 |    33060   |TainLi       |No
 |192.168.50.113 |    33070   |None         |Yes
 |192.168.50.113 |    33080   | HeJiaLi     |Yes
 |192.168.50.113 |    33090   |TangPing     |Yes
 |192.168.50.50  |    30320   | RenHui      |Yes
 |192.168.50.50  |    31010   | None        |Yes
 |192.168.50.50  |    31020   |Wenxiaobin(Scorbin)|Yes
 |192.168.50.50  |    31030   |Yangxiaodi   |No
 |192.168.50.50  |    31040   |Zouzhongquan|No
 |192.168.50.50  |    31050   |Hongxuesong  |No
 |192.168.50.50  |    31060   |HeJiaLi      |Yes
 

## Show tensorboard on your local machine
reference link[https://blog.csdn.net/bryant_meng/article/details/79153531]
## Stuck at apt-get update
Sometimes you may encounter :
0%[working...] when you update the source using command 
```
$ apt-get update
```
This problem probably stems from the directory: /etc/apt where apt-get update works
```
$ls /etc/apt/
apt.conf.d  auth.conf.d  preferences.d  sources.list source.list.d  trusted.gpg  trusted.gpg~
```
### SOLUTION:
* Find out the source you stuck when update ,for our cases,it's source.list.d,this directory stores additional source for some package.
```
$ rm -r /etc/apt/source.list.d
$ apt-get update
```





