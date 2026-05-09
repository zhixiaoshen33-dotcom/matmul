cmake_minimum_required(VERSION 3.16)

project(matmul_demo)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置交叉编译
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 指定编译器
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# ----------------------------
# Ascend 安装路径
# 根据你的实际安装位置修改
# ----------------------------
set(ASCEND_PATH /usr/local/Ascend)

# CANN toolkit
set(ASCEND_TOOLKIT_PATH
    ${ASCEND_PATH}/cann-8.5.0/aarch64-linux
)
file(GLOB_RECURSE ASCENDC_INCLUDE_DIRS
    ${ASCEND_TOOLKIT_PATH}/ascendc/include/*
)
# include
include_directories(
    ${ASCEND_TOOLKIT_PATH}/include
    ${ASCEND_TOOLKIT_PATH}/include/acl
    ${ASCEND_TOOLKIT_PATH}/ascendc/include
    ${ASCEND_TOOLKIT_PATH}/ascendc/include/basic_api
    ${ASCEND_TOOLKIT_PATH}/ascendc/include/basic_api/impl
    ${ASCEND_TOOLKIT_PATH}/include/ascendc
    ${ASCEND_TOOLKIT_PATH}/include/ascendc/basic_api
    ${ASCENDC_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# lib path
link_directories(
    ${ASCEND_TOOLKIT_PATH}/lib64
)

# ----------------------------
# host executable
# ----------------------------
add_executable(
    matmul_demo
    main.cpp
    matmul_kernel.cpp
)

# ----------------------------
# link acl runtime
# ----------------------------
target_link_libraries(
    matmul_demo
    acl
    ascendcl
    stdc++
)