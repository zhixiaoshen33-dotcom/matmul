#!/bin/bash
set -e

# 当前目录
CURRENT_DIR=$(cd "$(dirname "$0")"; pwd)

echo "Current dir: ${CURRENT_DIR}"

# 1. 加载 Ascend 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 如果你的环境路径不同，可以改成：
# source /usr/local/Ascend/latest/bin/setenv.bash

echo "Ascend environment loaded."

# 2. 清理旧编译文件
rm -rf build
mkdir -p build
cd build

# 3. CMake 编译
cmake ..

# 多线程编译
make -j$(nproc)

echo "Build success."

# 4. 返回工程目录
cd ${CURRENT_DIR}

# 5. 运行程序
echo "Start running on Ascend NPU..."

./build/matmul_demo

echo "Run finished."

## hello world :https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0004.html
