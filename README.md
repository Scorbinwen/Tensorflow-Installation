# Tutorial on logining the docker instance
Tensorflow installation tutorial for newbees...
* [Prerequisites](#prerequisites)
* [Tensorflow Installation](#Tensorflow-Installation)
* [Muti-GPUs synchronization](#Muti-GPUs-synchronization)
* [Others](#Others)
# Prerequisites
## Build your own docker instance
## Go to the GPU-management platform http://station.csgrandeur.com/gpu/faqs    
## Register using the inviting-code provided by @LiuNing
## Apply a port for accessing the servers.

# Tensorflow Installation

## Step0. CUDA Installation
* install CUDA Toolkit v8.0,or any other version you need(It has already been install on our Servers,you can skip this step.)            

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
```
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)
## Step1. install cuDNN v6.0
install the compatible(with CUDA) cuDNN version,for cuDNN v7 installation instruction,see [Muti-GPUs synchronization](#Muti-GPUs-synchronization)
```
$CUDNN_TAR_FILE="cudnn-8.0-linux-x64-v6.0.tgz"   
$wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/${CUDNN_TAR_FILE}   
$tar -xzvf ${CUDNN_TAR_FILE}   
$cp -P cuda/include/cudnn.h /usr/local/cuda-8.0/include    
$cp -P cuda/lib64/libcudnn* /usr/local/cuda-8.0/lib64/   
$chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*   
```
## Step2. set environment variables   
### add the 2 paths below to ~/.bashrc:   
```
$export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}    
$export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}    
```
[Explanation]: ${PATH:+:${PATH}} means that if PATH exists and PATH is not null then append the directory ${PATH} to PATH.

## Step3. download and install anaconda   
### select an compatible version of anaconda from "https://repo.continuum.io/archive/" or just use command below to download anaconda3:     
```
$wget https://repo.continuum.io/archive/Anaconda3-2.4.0-Linux-x86_64.sh    
```
## Step4. install anaconda    
```
$bash Anaconda3-2.4.0-Linux-x86_64.sh    
```
**Notice:   
Approve the licence at last and follow the installation navigation:     
Do you approve the license terms? [yes|no]    
[no] >>> yes**       
   

## Step5. download and install tensorflow-gpu
### add tsinghua conda source to accelerate the download speed.
```
$conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/    
$conda config --set show_channel_urls yes    

$conda create -n tensorflow-gpu python=3.6   
$source activate tensorflow-gpu    
```

**Notice:the command below may fail,just try more times.**        
```
$conda install tensorflow-gpu    
```
## Step6. update the pip source to accelerate the download speed
```
$source activate tensorflow-gpu    
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
Problems you may encounter:   
After you pip install opencv-python,you may still fail to import cv2:   
```
$python   
>>>import cv2    
Traceback (most recent call last):  
  File "<stdin>", line 1, in <module>   
  File "/root/anaconda3/envs/tensorflow-gpu/lib/python3.5/site-packages/cv2/__init__.py", line 3, in <module>   
    from .cv2 import *    
ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory   
```
[tips]:for problems(e.g.libgthread-2.0.so.0: cannot open shared object file: No such file or directory),you can search on Ubuntu forum
  
### Solve the problem above using command as follow:    
```
$apt-get update   
$apt-get -y upgrade     
$pip install opencv-python    
$apt-get install libgtk2.0-dev -y   
```

# Muti-GPUs synchronization
## Motivation
* Batch_Normalization Synchronization

## Requirement:      
### 1.cuda9 + cudnn-v7.1 + tensorflow-1.10+ + nccl:         
#### commands for install cudnn-v7.1:     
```
$wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz  
$tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz               
$cp cuda/include/cudnn.h /usr/local/cuda/include             
$cp cuda/lib64/libcudnn* /usr/local/cuda/lib64                     
$chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 
```
#### commands for uninstall nccl
```
$apt remove libnccl2 libnccl-dev
```
#### commands for install nccl
```
$dpkg -i nccl-repo-ubuntu1604-2.3.5-ga-cuda9.0_1-1_amd64.deb              
$apt update              
$apt install libnccl2=2.3.5-2+cuda9.0 libnccl-dev=2.3.5-2+cuda9.0             
```
#### commands to check whehter nccl is correctly installed:
```
$apt list | grep nccl
```
#### Using commands above,you may encounter error:
```
$/sbin/ldconfig.real: /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7 is not a symbolic link
```
#### Solve the problem above use the method below:
* check the link:                
```
$sudo ldconfig -v            
```

* Find the error:                  
/sbin/ldconfig.real: /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7 is not a symbolic link                   
libcudnn.so.7 -> libcudnn.so.7.0.5                 

* Create the new link manually:               
 ```
 $ln -sf /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7.0.5 /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudnn.so.7        
 ```
 [Note]: ln -s target symbolic-link
 
 # Others
 ## PS: use command "nvidia-smi" to monitor the usage of GPUs to see which processes that occupy the GPUs:  
```
$fuser -v /dev/nvidia*  
```
this command may fail:   
```
$fuser command not found  
```
### install the psmisc which contains command fuser:  
```
$apt-get install psmisc  
```
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
### Solution for the problem above
* Find out the source you stuck when update ,for our cases,it's source.list.d,this directory stores additional source for some package.
```
$ rm -r /etc/list/source.list.d
$ apt-get update
```
[Note]:     
The APT package index is essentially a database of available packages from the repositories defined in the /etc/apt/sources.list file and in the /etc/apt/sources.list.d directory.     
for more information about /etc/list,refer to [https://askubuntu.com/questions/82825/do-files-at-etc-apt-sources-list-d-need-to-have-an-extension-list]





