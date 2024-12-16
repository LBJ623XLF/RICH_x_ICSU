# RICH_x_ICSU

## DATA

### TOOL



### HUMAN

```bash
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="7idEzKSCpPKLTJwlUp3j")
project = rf.workspace("test-vongh").project("awkward-posture-of-human")
version = project.version(3)
dataset = version.download("yolov8")
```
### Camera
- **Model:** ZED

## Software Requirements

### Operating System
- **Ubuntu:** 20.04 (/ 22.04)
- ubuntu-20.04.6-desktop-amd64.iso
- https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/20.04/
- rufus-4.5.exe
- https://rufus.ie/zh/
- fat32 = 512MB, swap = 8GB, / = 12GB, /usr = 24GB, /home = 12GB, EFI = 512MB 

### Robotics Middleware
- **ROS1:** Noetic (/ 2 Humble)
- ubuntu?
- https://mirrors.tuna.tsinghua.edu.cn/

```bash
sudo apt update
cp /etc/apt/sources.list ~/Desktop 
sudo gedit /etc/apt/sources.list

# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
# deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
# deb-src http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```
