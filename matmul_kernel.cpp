#include "matmul.h"
#include "kernel_operator.h"

using namespace AscendC;

// host侧 main.cpp 会调用这个接口
extern "C" void matmul_kernel_do(
    void* d_A,
    void* d_B,
    void* d_C,
    int M,
    int N,
    int K
);

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}

    __aicore__ inline void Init(
        GM_ADDR a,
        GM_ADDR b,
        GM_ADDR c,
        int m,
        int n,
        int k
    )
    {
        M = m;
        N = n;
        K = k;

        // 绑定GM地址
        A_gm.SetGlobalBuffer(
            (half*)a,
            M * K
        );

        B_gm.SetGlobalBuffer(
            (half*)b,
            K * N
        );

        C_gm.SetGlobalBuffer(
            (float*)c,
            M * N
        );
    }

    __aicore__ inline void Process()
    {
        // -------------------------
        // 1. 从GM搬到Local Buffer
        // -------------------------
        LocalTensor<half> aLocal =
            inQueueA.AllocTensor<half>();

        LocalTensor<half> bLocal =
            inQueueB.AllocTensor<half>();

        DataCopy(
            aLocal,
            A_gm,
            M * K
        );

        DataCopy(
            bLocal,
            B_gm,
            K * N
        );

        // -------------------------
        // 2. 分配输出L0C
        // -------------------------
        LocalTensor<float> cLocal =
            outQueue.AllocTensor<float>();

        // -------------------------
        // 3. 构造matmul参数
        // -------------------------
        MMParam mmParam;

        mmParam.singleM = M;
        mmParam.singleN = N;
        mmParam.singleK = K;

        mmParam.isLeftTranspose = false;
        mmParam.isRightTranspose = false;

        // -------------------------
        // 4. 调用你 matmul.h 中实现
        // -------------------------
        MatmulFull<
            half,
            half,
            float,
            16,
            16,
            16,
            ABLayout::MK,
            ABLayout::KN
        >(
            aLocal,
            bLocal,
            aL1Buffer,
            bL1Buffer,
            cLocal,
            mmParam
        );

        // -------------------------
        // 5. 写回GM
        // -------------------------
        DataCopy(
            C_gm,
            cLocal,
            M * N
        );
    }

private:
    int M;
    int N;
    int K;

    // GM tensor
    GlobalTensor<half> A_gm;
    GlobalTensor<half> B_gm;
    GlobalTensor<float> C_gm;

    // queue
    TQue<QuePosition::VECIN, 1> inQueueA;
    TQue<QuePosition::VECIN, 1> inQueueB;
    TQue<QuePosition::VECOUT, 1> outQueue;

    // L1 buffer
    TBuf<TPosition::A1> aL1Buffer;
    TBuf<TPosition::B1> bL1Buffer;
};


// 真正的 Ascend kernel entry
extern "C" __global__ __aicore__
void matmul_kernel(
    GM_ADDR a,
    GM_ADDR b,
    GM_ADDR c,
    int M,
    int N,
    int K
)
{
    KernelMatmul op;

    op.Init(
        a,
        b,
        c,
        M,
        N,
        K
    );

    op.Process();
}


// host wrapper
extern "C"
void matmul_kernel_do(
    void* d_A,
    void* d_B,
    void* d_C,
    int M,
    int N,
    int K
)
{
    // 启动1个AI Core
    uint32_t blockDim = 1;

    matmul_kernel<<<blockDim>>>(
        (GM_ADDR)d_A,
        (GM_ADDR)d_B,
        (GM_ADDR)d_C,
        M,
        N,
        K
    );
}