## Overview
This codebase makes small modifications to the original MuJoCo MPC software application (https://github.com/deepmind/mujoco_mpc).

### Setup
We assume you are using an AWS EC2 c4.4xlarge instance.
You need to run this line each time you start a new session
`xvfb-run -a -s "-screen 0 1400x900x24" bash`

If you don't, you may seed this error
```
ERROR: could not initialize GLFW

Press Enter to exit ...
```

### Run an example
```
./build/bin/mjpc --task="Quadruped Flat" --steps=100 \
--horizon=0.35 --w0=0.0 --w1=0.0 --w2=0.0 --w3=0.0
```

The expected result should be
```
MuJoCo version 2.3.3
Hardware threads: 16
Agent threads: 13
```
There should also be a new video generated in `mujoco_mpc/videos`.

## Installation
The following should already be installed if you use the AMI image we provide,
but we leave these instructions in case something goes wrong.

### Get clang-14
```
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 14
```

### Get cmake
```
sudo apt install cmake
```

### Install MJPC packages
```
sudo apt-get install libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build libstdc++-12-dev
```

### Clone and build OpenCV
```
git clone https://github.com/opencv/opencv.git ~/opencv
cd ~/opencv
mkdir -p build && cd build
cmake ~/opencv \
  -D CMAKE_C_COMPILER=/usr/bin/clang-14 \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-14 \
  -D CMAKE_CXX_FLAGS="-stdlib=libstdc++" \
  -D CMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/12"
make -j12
sudo make install
```

### Build MJPC
From the `hw2/mujoco_mpc` directory, run:
```
cmake -Bbuild/ -Smjpc/ \
  -D CMAKE_C_COMPILER=/usr/bin/clang-14 \
  -D CMAKE_CXX_COMPILER=/usr/bin/clang++-14
cmake --build build/ --config Release --target mjpc -j 12 --
```

### Set up display
```
sudo apt install xvfb
xvfb-run -a -s "-screen 0 1400x900x24" bash
```
