#include <iostream>
#include <vector>
#include <acl/acl.h>

#define CHECK_ACL(cmd)                           \
    do {                                         \
        aclError ret = cmd;                      \
        if (ret != ACL_ERROR_NONE) {             \
            std::cout << "ACL Error: " << ret    \
                      << std::endl;              \
            exit(-1);                            \
        }                                        \
    } while (0)

// kernel函数声明（后面在 matmul_kernel.cpp 中实现）
extern "C" void matmul_kernel_do(
    void* d_A,
    void* d_B,
    void* d_C,
    int M,
    int N,
    int K
);

int main() {
    //-------------------------
    // 1. 定义矩阵维度
    //-------------------------
    int M = 2;
    int K = 3;
    int N = 2;

    // A: 2x3
    std::vector<float> h_A = {
        1,2,3,
        4,5,6
    };

    // B: 3x2
    std::vector<float> h_B = {
        1,2,
        3,4,
        5,6
    };

    // C: 2x2
    std::vector<float> h_C(M*N, 0);

    std::cout << "Input A:" << std::endl;
    for(auto x : h_A){
        std::cout << x << " ";
    }
    std::cout << std::endl;

    std::cout << "Input B:" << std::endl;
    for(auto x : h_B){
        std::cout << x << " ";
    }
    std::cout << std::endl;

    //-------------------------
    // 2. 初始化 Ascend runtime
    //-------------------------
    CHECK_ACL(aclInit(nullptr));

    int deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtContext context;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));

    //-------------------------
    // 3. Device申请GM内存
    //-------------------------
    float* d_A;
    float* d_B;
    float* d_C;

    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float);

    CHECK_ACL(
        aclrtMalloc(
            (void**)&d_A,
            sizeA,
            ACL_MEM_MALLOC_HUGE_FIRST
        )
    );

    CHECK_ACL(
        aclrtMalloc(
            (void**)&d_B,
            sizeB,
            ACL_MEM_MALLOC_HUGE_FIRST
        )
    );

    CHECK_ACL(
        aclrtMalloc(
            (void**)&d_C,
            sizeC,
            ACL_MEM_MALLOC_HUGE_FIRST
        )
    );

    //-------------------------
    // 4. Host -> Device
    //-------------------------
    CHECK_ACL(
        aclrtMemcpy(
            d_A,
            sizeA,
            h_A.data(),
            sizeA,
            ACL_MEMCPY_HOST_TO_DEVICE
        )
    );

    CHECK_ACL(
        aclrtMemcpy(
            d_B,
            sizeB,
            h_B.data(),
            sizeB,
            ACL_MEMCPY_HOST_TO_DEVICE
        )
    );

    //-------------------------
    // 5. 调用kernel
    //-------------------------
    matmul_kernel_do(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K
    );

    //-------------------------
    // 6. Device -> Host
    //-------------------------
    CHECK_ACL(
        aclrtMemcpy(
            h_C.data(),
            sizeC,
            d_C,
            sizeC,
            ACL_MEMCPY_DEVICE_TO_HOST
        )
    );

    //-------------------------
    // 7. 打印结果
    //-------------------------
    std::cout << "Output C:" << std::endl;

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout << h_C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    //-------------------------
    // 8. 释放资源
    //-------------------------
    aclrtFree(d_A);
    aclrtFree(d_B);
    aclrtFree(d_C);

    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
