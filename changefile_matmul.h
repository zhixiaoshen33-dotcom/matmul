/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 本程序为自由软件，您可以根据 CANN Open Software License Agreement Version 2.0（"许可证"）
 * 的条款和条件对其进行再分发和/或修改。
 * 请参阅许可证了解详情。未经许可证授权，不得使用本文件。
 * 本软件按"原样"提供，不附带任何形式的明示或暗示保证，
 * 包括但不限于不侵权、适销性或特定用途适用性的保证。
 * 完整许可证文本请参见软件仓库根目录的 LICENSE 文件。
 */

/*!
 * \file matmul.h
 * \brief 华为昇腾 AscendC 矩阵乘法（Matmul）核心库
 *
 * 本文件实现了在昇腾 AI 处理器（Ascend NPU）上执行高性能矩阵乘法的
 * 一系列模板函数，覆盖以下场景：
 *
 *  ┌─────────────────────────────────────────────────────────┐
 *  │  函数名         │  分块策略         │  适用场景          │
 *  ├─────────────────────────────────────────────────────────┤
 *  │  MatmulFull     │  全矩阵一次性载入  │  小矩阵，M/K/N均小 │
 *  │  MatmulK        │  沿 K 轴分块      │  K 维度很大        │
 *  │  MatmulN        │  沿 N 轴分块      │  N 维度很大        │
 *  │  MatmulKM       │  沿 K 和 M 分块   │  K、M 均很大       │
 *  │  MatmulKPP      │  K 轴流水分块     │  追求最高吞吐      │
 *  │  MatmulFullMX   │  全载 MX 量化版   │  FP8 微缩量化      │
 *  │  MatmulKMx      │  K 分块 MX 量化   │  FP8 + 大 K        │
 *  └─────────────────────────────────────────────────────────┘
 *
 * 昇腾内存层次结构（从慢到快）：
 *   GM（全局内存）→ L1（片上一级缓存）→ L0A/L0B（矩阵运算缓冲）→ L0C（累加结果缓冲）
 *
 * 核心硬件指令：
 *   - LoadData / LoadData3D：将数据从 L1 搬运到 L0A 或 L0B，并完成分形格式转换
 *   - Mmad：矩阵乘累加指令，在 AI Core 的 Cube 单元上执行
 *
 * 支持的数据类型组合（A类型 × B类型 → C类型）：
 *   half × half → float     （最常用）
 *   float × float → float
 *   int8 × int8 → int32
 *   fp8_e4m3 × fp8_e4m3 → float   （量化推理）
 *   fp8_e5m2 × fp8_e5m2 → float   （量化推理）
 *   mx_fp8_e4m3 × mx_fp8_e4m3 → float  （MX 微缩量化）
 */

#ifndef MATMUL_H
#define MATMUL_H

// #include "buffers_policy.h"   // 包含 AscendC 缓冲区策略定义（L0A/L0B/L0C 双缓冲等）
#include "kernel_operator.h"

using namespace AscendC;      // 使用 AscendC 命名空间，包含 LocalTensor、GlobalTensor、Mmad 等核心 API

namespace fa_base_matmul {    // 本库的私有命名空间，避免与用户代码冲突

// ============================================================
// 全局常量定义
// ============================================================

// Mmad unitFlag 控制字段：控制矩阵乘法单元的行为标志
constexpr uint32_t UNITFLAG_DISABLE      = 0;  // 不使用 unitFlag（普通模式）
constexpr uint32_t UNITFLAG_ENABLE       = 2;  // 启用 unitFlag（切 K 中间轮次）
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3; // 启用 unitFlag，且这是外层 K 循环的最后一轮

// 昇腾 NPU 硬件的"分形"（Fractal）是矩阵数据的基本存储单元
// 每个分形在物理上占 512 字节
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16; // fp16 分形：16×16 个元素 = 512B
static constexpr uint32_t INT4_ONE_FRACTAL_ELEMENT = 64; // int4 分形：16×64 个元素 = 512B
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT    = 16; // 分形在 height 方向的元素数
static constexpr uint32_t ONE_FRACTAL_W_BYTE       = 32; // 分形在 width 方向的字节数（32B）

static constexpr uint32_t LOAD3D_L1W_SIZE  = 16; // Load3D 指令的 L1 width 参数
static constexpr uint32_t MMAD_MN_SIZE_10  = 10; // Mmad 特殊情况：MN 尺寸标志

// Load3D 卷积参数（用于 MK 布局的矩阵加载，将矩阵乘法映射为卷积操作）
static constexpr uint8_t LOAD3D_STRIDE_W       = 1; // 卷积核在 width 方向的滑动步长
static constexpr uint8_t LOAD3D_STRIDE_H       = 1; // 卷积核在 height 方向的滑动步长
static constexpr uint8_t LOAD3D_FILTER_W       = 1; // 卷积核 width 大小
static constexpr uint8_t LOAD3D_FILTER_H       = 1; // 卷积核 height 大小
static constexpr uint8_t LOAD3D_DILA_FILTER_W  = 1; // 卷积核 width 膨胀系数
static constexpr uint8_t LOAD3D_DILA_FILTER_H  = 1; // 卷积核 height 膨胀系数

// 对齐基数（某些数据类型要求额外的对齐）
static constexpr uint32_t K_STEP_ALIGN_BASE = 2; // K 轴步长对齐基数
static constexpr uint32_t M_STEP_ALIGN_BASE = 2; // M 轴步长对齐基数（fp8 转置场景）

// ============================================================
// MMParam：矩阵乘法参数结构体
// ============================================================
/**
 * @brief 矩阵乘法参数，传递给所有 Matmul* 函数
 *
 * 计算：C(M×N) = A(M×K) × B(K×N)
 * 其中 A、B 可以各自选择是否转置存储。
 */
struct MMParam {
    uint32_t singleM;          ///< 本次计算的 M 维度大小（行数）
    uint32_t singleN;          ///< 本次计算的 N 维度大小（列数）
    uint32_t singleK;          ///< 本次计算的 K 维度大小（内积维度）

    bool isLeftTranspose;      ///< A 矩阵是否以转置格式存储在 L1（KM 布局时为 true）
    bool isRightTranspose;     ///< B 矩阵是否以转置格式存储在 L1（NK 布局时为 true）

    bool cmatrixInitVal = true; ///< true：首轮 K 计算时对 C（L0C）清零并初始化
                                ///< false：累加到已有的 C 值上（多轮 K 累加时使用）

    bool isOutKFisrt = true;   ///< 在外层 K 分块场景中，标记是否为第一轮 K
                               ///< true（默认）= 首轮，L0C 需要初始化；false = 中间轮/尾轮，累加

    uint32_t unitFlag = 0;     ///< Mmad 单元标志，控制多轮累加的边界行为：
                               ///<   0（DISABLE）：不设置，普通矩阵乘
                               ///<   2（ENABLE）：切 K 的中间轮次
                               ///<   3（EN_OUTER_LAST）：外层 K 循环的最后一轮，触发最终写回

    uint32_t realM = 0;        ///< 实际有效的 M 大小（用于 BMM2 等场景，s1realsize≠singleM 时使用）
                               ///< 为 0 时不影响现有逻辑，即以 singleM 为准
};

/**
 * @brief MMParam 的工厂函数，用于方便地构造参数结构体
 *
 * 示例用法：
 *   auto param = MakeMMParam(32, 32, 32, false, false);
 *   // 表示 C(32×32) = A(32×32,MK布局) × B(32×32,KN布局)，首轮K，不设unitFlag
 */
__aicore__ inline MMParam MakeMMParam(
    uint32_t singleM, uint32_t singleN, uint32_t singleK,
    bool isLeftTranspose, bool isRightTranspose,
    bool cmatrixInitVal = true, bool isOutKFisrt = true,
    uint32_t unitFlag = 0, uint32_t realM = 0)
{
    return {
        .singleM         = singleM,
        .singleN         = singleN,
        .singleK         = singleK,
        .isLeftTranspose  = isLeftTranspose,
        .isRightTranspose = isRightTranspose,
        .cmatrixInitVal  = cmatrixInitVal,
        .isOutKFisrt     = isOutKFisrt,
        .unitFlag        = unitFlag,
        .realM           = realM
    };
}

// ============================================================
// ABLayout：矩阵在 L1 中的存储布局枚举
// ============================================================
/**
 * @brief 描述 A 矩阵或 B 矩阵在 L1 缓冲区中的维度排列方式
 *
 * 对于 A 矩阵（M×K）：
 *   MK = 行主序（最常见），即 A[m][k] 在内存中按 m 优先排列
 *   KM = 列主序（等价于 A 已被转置），即 A^T[k][m]
 *
 * 对于 B 矩阵（K×N）：
 *   KN = 行主序（最常见），即 B[k][n]
 *   NK = 列主序（等价于 B 已被转置），即 B^T[n][k]
 */
enum class ABLayout {
    MK = 0,  ///< A 矩阵：行主序（非转置）
    KM = 1,  ///< A 矩阵：列主序（已转置）
    KN = 2,  ///< B 矩阵：行主序（非转置）
    NK = 3,  ///< B 矩阵：列主序（已转置）
};

// ============================================================
// AlignUp：向上对齐工具函数
// ============================================================
/**
 * @brief 将 num 向上对齐到 rnd 的整数倍
 * @param num  需要对齐的数值
 * @param rnd  对齐基数
 * @return     ≥ num 的最小 rnd 整数倍
 *
 * 示例：AlignUp(13, 16) = 16；AlignUp(32, 16) = 32
 */
template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

// ============================================================
// 以下代码针对昇腾 310/310R6 及架构5102 系列芯片
// ============================================================
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))

/**
 * @brief 计算给定元素数量在 L0 中所需的"块数"（以32B为单位）
 *
 * Ascend 的 LoadData2D 指令以 32B（kStep）为单位描述 K 轴长度。
 * 不同数据类型的对齐方式不同：
 *   - float（fp32）：8 个元素 = 32B，向上对齐到 8 的倍数后 ÷8
 *   - fp8/int8：32 个元素 = 32B，向上对齐到 32 的倍数后 ÷32
 *   - half（fp16）/ 其他：16 个元素 = 32B，向上对齐到 16 的倍数后 ÷16
 *
 * @tparam T  数据类型
 * @param  size  元素数量
 * @return  以 32B 为单位的块数（即 LoadData2D 的 kStep 参数）
 */
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size) {
    if constexpr (IsSameType<T, float>::value) {
        // float：每 8 个元素 = 32B，向上对齐 8 再除 8
        return ((size + 7) >> 3 << 3) >> 3;
    } else if constexpr ((IsSameType<T, fp8_e5m2_t>::value ||
                          IsSameType<T, fp8_e4m3fn_t>::value ||
                          IsSameType<T, hifloat8_t>::value ||
                          IsSameType<T, int8_t>::value)) {
        // fp8/int8：每 32 个元素 = 32B，向上对齐 32 再除 32
        return ((size + 31) >> 5 << 5) >> 5;
    } else {
        // half（fp16）：每 16 个元素 = 32B，向上对齐 16 再除 16
        return ((size + 15) >> 4 << 4) >> 4;
    }
}

// ============================================================
// LoadDataToL0A：将 A 矩阵数据从 L1 搬运到 L0A
// ============================================================
/**
 * @brief 把 A 矩阵的一个分块从 L1 缓冲区搬运到 L0A，并完成分形格式转换
 *
 * 昇腾的矩阵乘法要求输入数据必须以"分形格式"存储在 L0A/L0B 中。
 * LoadData 指令负责将 L1 中的普通行主序/列主序数据自动转换为分形格式。
 *
 * 参数说明（以 A 矩阵 M×K 为例）：
 * @tparam T          数据类型（half、float、fp8_*、int8_t 等）
 * @param  aL0Tensor  目标：L0A 缓冲区的 LocalTensor（分形格式）
 * @param  aL1Tensor  源：L1 缓冲区中的 A 矩阵数据
 * @param  mmParam    矩阵乘法参数（含 isLeftTranspose、singleM/K/N 等）
 * @param  L1Aoffset  A 矩阵在 L1 中的起始偏移（元素数，用于 K 分块寻址）
 * @param  kSplitSize 本次要搬运的 K 维度大小
 * @param  mSplitSize 本次要搬运的 M 维度大小
 *
 * 内部逻辑：
 *   - 若 isLeftTranspose=true（KM 布局）：A 以列主序存储，LoadData 转置加载
 *   - 若 isLeftTranspose=false（MK 布局）：A 以行主序存储，直接加载
 *   - 对 fp8/int8 的转置场景，因为 mStep 必须为偶数对齐，需拆分为多次 LoadData
 */
template <typename T>
__aicore__ inline void LoadDataToL0A(
    LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
    const MMParam& mmParam, uint64_t L1Aoffset,
    uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // LoadData2D 指令的参数结构体

    // 源矩阵 M 轴方向的起始位置（单位：16个元素，即一个分形的 height）
    loadData2DParamsA.mStartPosition = 0;
    // 源矩阵 K 轴方向的起始位置（单位：32B，即一个分形的 width 字节数）
    loadData2DParamsA.kStartPosition = 0;
    // 是否对每个分形矩阵执行转置（KM布局时需要转置加载到L0A）
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose;

    if (loadData2DParamsA.ifTranspose) {
        // KM 布局（A 已转置）：L1 中实际存储的是 K×M，
        // LoadData 的 mStep 描述"源矩阵 K 轴方向搬运长度"（转置后视角下的行数）
        // 向上对齐到 16 再除以 16，单位：16个元素
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4;

        // fp8/int8 要求 mStep 为偶数（硬件对齐约束）
        if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                      IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value ||
                      IsSameType<T, int8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }

        // kStep：转置场景下，描述 M 轴方向的块数（单位：32B）
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize);
    } else {
        // MK 布局（正常行主序）：mStep 描述 M 轴搬运长度（行数÷16）
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
        // kStep 描述 K 轴长度（字节块数）
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize);
    }

    // float 类型转置时，kStep 需要额外的 K_STEP_ALIGN_BASE 对齐
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsA.ifTranspose) {
            loadData2DParamsA.kStep = CeilAlign(loadData2DParamsA.kStep, K_STEP_ALIGN_BASE);
        }
    }

    // srcStride：源矩阵中相邻两个分形（沿 K 方向）之间的地址间隔（单位：512B）
    // 对于 fp8/int8：需要 64 元素对齐（配合 UB→L1 的搬运对齐要求）
    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value ||
                  IsSameType<T, int8_t>::value) {
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose
            ? ((kSplitSize + 63) >> 6 << 6) >> 4   // 转置：按 64 对齐后 ÷16（单位512B）
            : ((mSplitSize + 31) >> 5 << 5) >> 4;  // 非转置：按 32 对齐
    } else {
        // half/float 普通情况
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose
            ? ((mmParam.singleK + 15) >> 4 << 4) >> 4  // 转置：整行 K 维度的分形数
            : loadData2DParamsA.mStep;                  // 非转置：等于 mStep
    }

    // realM 覆盖：BMM2 等场景下，实际 M 大小可能小于 singleM
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }

    // dstStride：目标（L0A）中相邻两个分形之间的地址间隔（单位：512B）
    // 转置时：dstStride = ceil(mSplitSize / 16)，非转置时与 mStep 相同
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose
        ? (mSplitSize + 15) >> 4
        : loadData2DParamsA.mStep;

    // fp8/int8 转置时，mStep 必须为 2 的倍数，若超出则拆分为多次 LoadData
    // 每次 LoadData 处理 2 行（M_STEP_ALIGN_BASE = 2），地址按 dstAddrStride 递增
    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value ||
                  IsSameType<T, int8_t>::value) {
        if (loadData2DParamsA.ifTranspose) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1; // 需要执行的 LoadData 次数
            loadData2DParamsA.mStep = M_STEP_ALIGN_BASE;             // 每次搬运 2 行
            uint64_t dstOffset    = 0;
            // 每次 LoadData 写入 L0A 的偏移量增量（单位：元素数）
            uint64_t dstAddrStride = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsA.mStartPosition;

            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset], loadData2DParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
        }
    } else {
        // half/float：直接一次 LoadData 完成
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

// ============================================================
// LoadDataToL0AMx：MX 微缩量化格式的 A 矩阵加载（L1→L0A）
// ============================================================
/**
 * @brief MX（MicroScaling）量化格式的 A 矩阵加载
 *
 * MX 格式在每个数据元素旁附带一个 fp8_e8m0_t 类型的缩放因子（scale），
 * 用于微缩量化推理（如 FP8 × FP8 矩阵乘法）。
 * 此函数同时搬运数据（aL1Tensor）和对应的缩放因子（aScaleL1Tensor）到 L0A。
 *
 * @tparam T          输入数据类型（fp8_e4m3fn_t 等量化类型）
 * @tparam U          L0A 目标类型（mx_fp8_e4m3_t 等 MX 格式类型）
 * @param  aScaleL1Tensor  A 矩阵的缩放因子，存储在 L1 中，类型为 fp8_e8m0_t
 * （其他参数同 LoadDataToL0A）
 */
template <typename T, typename U = T>
__aicore__ inline void LoadDataToL0AMx(
    LocalTensor<U>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
    const LocalTensor<fp8_e8m0_t>& aScaleL1Tensor,
    const MMParam& mmParam, uint64_t L1Aoffset,
    uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA;
    loadData2DParamsA.mStartPosition = 0;
    loadData2DParamsA.kStartPosition = 0;
    loadData2DParamsA.ifTranspose    = mmParam.isLeftTranspose;

    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4;
        // fp8 类型要求 mStep 偶数对齐
        if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                      IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep = GetBlockNum<T>(mSplitSize);
    } else {
        loadData2DParamsA.mStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsA.kStep = GetBlockNum<T>(kSplitSize);
    }

    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value) {
        // fp8 转置：srcStride 按 256 对齐（配合 256×32/256 对齐要求）
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose
            ? 256 >> 4                             // 固定 256B 间隔 ÷16
            : ((mSplitSize + 31) >> 5 << 5) >> 4; // 非转置：32 元素对齐
    } else {
        loadData2DParamsA.srcStride = loadData2DParamsA.ifTranspose
            ? ((mmParam.singleK + 15) >> 4 << 4) >> 4
            : loadData2DParamsA.mStep;
    }

    // MX scale 数据的加载参数（LoadData2DMxParams）
    // scale 按 64 元素（32B×2）分组，与数据分形一一对应
    LoadData2DMxParams loadData2DMxParamsA;
    loadData2DMxParamsA.xStartPosition = 0;                          // scale 在 M 方向的起始位置
    loadData2DMxParamsA.yStartPosition = 0;                          // scale 在 K 方向的起始位置
    loadData2DMxParamsA.xStep = ((mSplitSize + 15) >> 4 << 4) >> 4; // M 方向 scale 步长
    loadData2DMxParamsA.yStep = (kSplitSize + 63) / 32 / 2;         // K 方向 scale 步长（每组64元素，32B/组）
    loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;       // 源 scale 的行间隔
    loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;       // 目标 L0A 中 scale 的行间隔

    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep      = ((mmParam.realM + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.xStep    = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }

    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose
        ? (mSplitSize + 15) >> 4
        : loadData2DParamsA.mStep;

    // fp8 转置且 dstStride 为奇数时，需拆分为多次 LoadData（同 LoadDataToL0A 中的处理）
    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value) {
        if (loadData2DParamsA.ifTranspose && (loadData2DParamsA.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1;
            loadData2DParamsA.mStep      = M_STEP_ALIGN_BASE;
            loadData2DMxParamsA.xStep    = loadData2DParamsA.mStep;
            uint64_t dstOffset           = 0;
            uint64_t dstAddrStride       = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep            = loadData2DParamsA.mStartPosition;
            uint16_t oriMScaleStep       = loadData2DMxParamsA.xStartPosition;

            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition    = oriMStep + M_STEP_ALIGN_BASE * idx;
                loadData2DMxParamsA.xStartPosition  = oriMScaleStep + M_STEP_ALIGN_BASE * idx;
                // 同时搬运数据和 scale（MX LoadData 的扩展版本）
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset],
                         aScaleL1Tensor[L1Aoffset / 32],
                         loadData2DParamsA, loadData2DMxParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset],
                     aScaleL1Tensor[L1Aoffset / 32],
                     loadData2DParamsA, loadData2DMxParamsA);
        }
    } else {
        // 非 fp8 类型：退化为普通 LoadData（忽略 scale）
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

// ============================================================
// LoadDataToL0B：将 B 矩阵数据从 L1 搬运到 L0B
// ============================================================
/**
 * @brief 把 B 矩阵的一个分块从 L1 缓冲区搬运到 L0B，并完成分形格式转换
 *
 * B 矩阵的加载逻辑与 A 矩阵对称，但有一个关键差异：
 *   B 矩阵的 LoadData 转置标志是 isRightTranspose 的"取反"。
 *   原因：Mmad 指令要求 L0B 必须以 KN 格式（K 轴连续）存储；
 *   若 B 以 KN 布局（非转置）存储在 L1 中，LoadData 需要"转置加载"来满足 L0B 的要求；
 *   若 B 以 NK 布局（已转置）存储在 L1 中，LoadData 反而直接加载即可。
 *
 * @param nLoops  N 轴循环次数（保留参数，某些流水线场景使用）
 * （其他参数同 LoadDataToL0A）
 */
template <typename T>
__aicore__ inline void LoadDataToL0B(
    LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
    const MMParam& mmParam, uint64_t L1Boffset,
    uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB;
    loadData2DParamsB.mStartPosition = 0;
    loadData2DParamsB.kStartPosition = 0;
    // 情况1：isRightTranspose = false，B 在 L1 以 KN 格式存储
    //    KN → 需要转置 → NK（L0B要求）
    //    ifTranspose = true = !false ✓

    // 情况2：isRightTranspose = true，B 在 L1 以 NK 格式存储
    //     NK → 已经是目标格式，不需要转置
        ifTranspose = false = !true ✓
    // 注意：B 矩阵的 ifTranspose 是 isRightTranspose 的"非"
    // KN布局（非转置存储）→ 需要转置加载 → ifTranspose=true
    // NK布局（已转置存储）→ 直接加载     → ifTranspose=false
    loadData2DParamsB.ifTranspose = !mmParam.isRightTranspose;

    if (loadData2DParamsB.ifTranspose) {
        // B 以 KN 布局存储（K 行 N 列），L0B 需要 NK 格式
        // mStep：K 轴方向的行数（单位：16 元素）
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4;
        if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                      IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value ||
                      IsSameType<T, int8_t>::value) {
            loadData2DParamsB.mStep = (loadData2DParamsB.mStep + 1) >> 1 << 1;
        }
        // kStep：N 轴方向的块数（单位：32B）
        loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize);
    } else {
        // B 以 NK 布局存储（N 行 K 列）
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize);
    }

    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.kStep = CeilAlign(loadData2DParamsB.kStep, K_STEP_ALIGN_BASE);
        }
    }

    // srcStride：源矩阵（L1 中的 B）相邻分形之间的地址间隔（单位：512B）
    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value ||
                  IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4; // K 方向：32 对齐
        } else {
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4; // N 方向：32 对齐
        }
    } else {
        // half/float：以整行宽度作为 stride
        loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose
            ? (((mmParam.singleK + 15) >> 4 << 4) >> 4)    // 转置：整行 K 的分形数
            : (((mmParam.singleN + 15) >> 4 << 4) >> 4);   // 非转置：整行 N 的分形数
    }

    // dstStride：目标（L0B）中相邻分形的地址间隔
    loadData2DParamsB.dstStride = loadData2DParamsB.ifTranspose
        ? (nSplitSize + 15) >> 4     // 转置时：目标中 N 方向的分形数
        : loadData2DParamsB.mStep;   // 非转置时：等于 mStep

    // fp8/int8 转置场景的多次拆分加载（同 LoadDataToL0A 的处理逻辑）
    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value ||
                  IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset      = 0;
            uint64_t dstAddrStride  = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep       = loadData2DParamsB.mStartPosition;

            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], loadData2DParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
        }
    } else {
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    }
}

// ============================================================
// LoadDataToL0BMx：MX 量化格式的 B 矩阵加载（L1→L0B）
// ============================================================
/**
 * @brief MX 格式的 B 矩阵加载，同时搬运数据和 scale 因子
 * 逻辑与 LoadDataToL0AMx 对称，作用于 B 矩阵和 L0B 缓冲区。
 */
template <typename T, typename U = T>
__aicore__ inline void LoadDataToL0BMx(
    LocalTensor<U>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
    const LocalTensor<fp8_e8m0_t>& bScaleL1Tensor,
    const MMParam& mmParam, uint64_t L1Boffset,
    uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB;
    loadData2DParamsB.mStartPosition = 0;
    loadData2DParamsB.kStartPosition = 0;
    loadData2DParamsB.ifTranspose    = !mmParam.isRightTranspose; // 与 LoadDataToL0B 相同的取反逻辑

    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsB.kStep = GetBlockNum<T>(nSplitSize);
    } else {
        loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DParamsB.kStep = GetBlockNum<T>(kSplitSize);
    }

    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value) {
        loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose
            ? ((kSplitSize + 31) >> 5 << 5) >> 4
            : ((nSplitSize + 31) >> 5 << 5) >> 4;
    } else {
        loadData2DParamsB.srcStride = loadData2DParamsB.ifTranspose
            ? (((mmParam.singleK + 15) >> 4 << 4) >> 4)
            : (((mmParam.singleN + 15) >> 4 << 4) >> 4);
    }

    loadData2DParamsB.dstStride = loadData2DParamsB.ifTranspose
        ? (nSplitSize + 15) >> 4
        : loadData2DParamsB.mStep;

    // B 矩阵的 MX scale 参数配置（与 A 对称）
    LoadData2DMxParams loadData2DMxParamsB;
    loadData2DMxParamsB.xStartPosition = 0;
    loadData2DMxParamsB.yStartPosition = 0;
    loadData2DMxParamsB.xStep     = ((nSplitSize + 15) >> 4 << 4) >> 4; // N 方向 scale 步长
    loadData2DMxParamsB.yStep     = (kSplitSize + 63) / 32 / 2;         // K 方向 scale 步长
    loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
    loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;

    if constexpr (IsSameType<T, fp8_e5m2_t>::value ||
                  IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value) {
        if (loadData2DParamsB.ifTranspose && (loadData2DParamsB.dstStride & 1)) {
            // 拆分为多次 LoadData（同 LoadDataToL0AMx 的处理）
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep      = M_STEP_ALIGN_BASE;
            loadData2DMxParamsB.xStep    = loadData2DParamsB.mStep;
            uint64_t dstOffset           = 0;
            uint64_t dstAddrStride       = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep            = loadData2DParamsB.mStartPosition;
            uint16_t oriMScaleStep       = loadData2DMxParamsB.xStartPosition;

            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition    = oriMStep + M_STEP_ALIGN_BASE * idx;
                loadData2DMxParamsB.xStartPosition  = oriMScaleStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset],
                         bScaleL1Tensor[L1Boffset / 32],
                         loadData2DParamsB, loadData2DMxParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset],
                     bScaleL1Tensor[L1Boffset / 32],
                     loadData2DParamsB, loadData2DMxParamsB);
        }
    } else {
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    }
}

// ============================================================
// MatmulFullMX：全载 MX 量化矩阵乘（310/310R6/5102 架构）
// ============================================================
/**
 * @brief A 和 B 矩阵整体装载到 L0A/L0B 后执行 Mmad（MX 量化版本）
 *
 * "Full" 表示不在 K/M/N 方向分块，一次性把整个 A 和 B 载入 L0。
 * 适用于矩阵尺寸较小（能够完整放入 L0A+L0B）的情况。
 *
 * 执行流程：
 *   1. 等待 L0A 空闲（上一轮 Mmad 完成）
 *   2. LoadData：A 从 L1 → L0A（分形格式转换）
 *   3. 通知 L0A 数据就绪
 *   4. 等待 L0B 空闲，LoadData：B 从 L1 → L0B
 *   5. 等待 L0A、L0B 均就绪
 *   6. Mmad：L0A × L0B → L0C
 *   7. 释放 L0A、L0B（可供下一轮使用）
 *
 * @tparam A       A 矩阵数据类型
 * @tparam B       B 矩阵数据类型
 * @tparam C       C 矩阵数据类型（输出，通常为 float 或 int32）
 * @tparam baseM   M 方向的分形基本单位（通常为 16）
 * @tparam baseN   N 方向的分形基本单位（通常为 16）
 * @tparam baseK   K 方向的分形基本单位（通常为 16 或 32）
 * @tparam AL      A 矩阵布局（MK 或 KM）
 * @tparam BL      B 矩阵布局（KN 或 NK）
 * @tparam L0AType L0A 双缓冲对象类型
 * @tparam L0BType L0B 双缓冲对象类型
 * @tparam AScaleType A 矩阵 scale 类型（MX 量化用，默认 float）
 * @tparam BScaleType B 矩阵 scale 类型（MX 量化用，默认 float）
 * @tparam L0ADType L0A 中实际存储的数据类型（MX 场景可能与 A 不同）
 * @tparam L0BDType L0B 中实际存储的数据类型
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType,
          typename AScaleType = float, typename BScaleType = float,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFullMX(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<AScaleType>& aScaleL1Tensor,
    const LocalTensor<BScaleType>& bScaleL1Tensor,
    L0AType& aL0BuffsDb,
    L0BType& bL0BuffsDb,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    // 步骤1：从双缓冲池中获取一个 L0A 缓冲区
    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    // 等待 Mmad 单元释放（即上一轮计算已完成，L0A 可以写入新数据）
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();

    // 步骤2：将 A 从 L1 搬运到 L0A（根据 L0A 类型选择普通或 MX 版本）
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        // MX FP8 格式：需要同时搬运数据和 scale
        LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor,
                                     param, 0, param.singleK, param.singleM);
    } else if constexpr (IsSameType<L0ADType, fp8_e4m3fn_t>::value) {
        // 普通 FP8 格式
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    }

    // 步骤3：通知 Mmad 单元：L0A 数据已就绪，可以开始计算
    l0aBuffer.Set<HardEvent::MTE1_M>();

    // 步骤4-5：同理加载 B 到 L0B
    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();

    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor,
                                     param, 0, param.singleK, param.singleN);
    } else if constexpr (IsSameType<L0BDType, fp8_e4m3fn_t>::value) {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }

    l0bBuffer.Set<HardEvent::MTE1_M>();

    // 步骤6：等待 L0A 和 L0B 均就绪后，执行 Mmad
    l0aBuffer.Wait<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m             = param.singleM;
    if (param.realM != 0) {
        mmadParams.m         = param.realM; // 使用实际 M 大小（BMM2 等特殊场景）
    }
    mmadParams.n             = param.singleN;
    mmadParams.k             = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt; // 首轮 K 时清零 L0C
    mmadParams.cmatrixSource  = false;              // 不从外部读取初始 C 值
    mmadParams.unitFlag       = param.unitFlag;

    // m=1 时硬件会进入 GEMV 模式，规避方法：强制设为 16
    if (mmadParams.m == 1) { mmadParams.m = 16; }

    // 步骤7：执行矩阵乘累加，结果写入 L0C
    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    // 步骤8：释放 L0A、L0B，通知 MTE1 可以搬运下一批数据
    l0aBuffer.Set<HardEvent::M_MTE1>();
    l0bBuffer.Set<HardEvent::M_MTE1>();
}

// ============================================================
// MatmulKMx：沿 K 轴分块的 MX 量化矩阵乘（310/310R6/5102 架构）
// ============================================================
/**
 * @brief 将 K 维度切分为多个 baseK 大小的块，逐块执行 Mmad 并累加到 L0C
 *
 * 适用于 K 维度较大（无法整体装入 L0A）的情况。
 * 核心思想：
 *   for k in range(0, K, baseK):
 *       LoadData A[m, k:k+baseK] → L0A
 *       LoadData B[k:k+baseK, n] → L0B
 *       Mmad(L0A, L0B) 累加到 L0C
 *
 * 硬件事件同步（HardEvent）：
 *   - M_MTE1：Mmad 通知 MTE1（搬运单元）可以开始加载下一块数据
 *   - MTE1_M：MTE1 通知 Mmad 数据加载完成，可以开始计算
 * 这构成了一个"双缓冲流水线"，使搬运和计算可以并行进行。
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType,
          typename AScaleType = float, typename BScaleType = float,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulKMx(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<AScaleType>& aScaleL1Tensor,
    const LocalTensor<BScaleType>& bScaleL1Tensor,
    L0AType& aL0BuffsDb,
    L0BType& bL0BuffsDb,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    // 计算 K 方向的循环次数和尾块大小
    uint32_t kLoops   = (param.singleK + baseK - 1) / baseK; // 总块数（上取整）
    uint32_t tailSize = param.singleK % baseK;                 // 最后一块的实际大小
    uint32_t tailK    = tailSize ? tailSize : baseK;           // 尾块（若整除则等于 baseK）

    // 计算 A、B 在 L1 中每一块的地址步长（字节偏移）
    // 对于 A（MK 布局，非转置）：每块沿 K 方向移动 M_rows * baseK 个元素
    // 对于 A（KM 布局，转置）：每块沿 K 方向移动 baseK * 16（一个分形宽度）
    uint64_t L1Aoffset = param.isLeftTranspose
        ? baseK << 4                                           // 转置：baseK×16 元素
        : ((param.singleM + 15) >> 4 << 4) * baseK;           // 非转置：M 对齐后 × baseK

    uint64_t L1Boffset = param.isRightTranspose
        ? ((param.singleN + 15) >> 4 << 4) * baseK            // 转置：N 对齐后 × baseK
        : baseK << 4;                                          // 非转置：baseK×16

    // 310/310R6 芯片对 fp8/int8 有特殊的对齐要求（32 元素对齐）
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value ||
                  IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float>::value) {
        // float 转置时的地址步长不同（float 每个分形 width 为 8 元素而非 16）
        L1Aoffset = param.isLeftTranspose
            ? baseK << 3
            : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose
            ? ((param.singleN + 15) >> 4 << 4) * baseK
            : baseK << 3;
    }
#endif

    // 主循环：逐块加载并计算
    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK; // 当前块的 K 大小

        // ---- 加载 A 的第 k 块到 L0A ----
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // 等待上一轮 Mmad 完成后才能写入新数据
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();

        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor,
                                         param, k * L1Aoffset, tileK, param.singleM);
        } else if constexpr (IsSameType<L0ADType, fp8_e4m3fn_t>::value) {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        }
        l0aBuffer.Set<HardEvent::MTE1_M>(); // 通知 Mmad：A 数据就绪

        // ---- 加载 B 的第 k 块到 L0B ----
        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();

        // B 的 NK 布局（转置存储）不需要沿 K 方向移动指针（所有 K 数据在同一位置）
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;

        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor,
                                         param, k * L1Boffset, tileK, param.singleN, loopNum);
        } else if constexpr (IsSameType<L0BDType, fp8_e4m3fn_t>::value) {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        }
        l0bBuffer.Set<HardEvent::MTE1_M>();

        // ---- 等待双缓冲均就绪，执行 Mmad ----
        l0aBuffer.Wait<HardEvent::MTE1_M>();
        l0bBuffer.Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) { mmadParams.m = param.realM; }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) { mmadParams.m = 16; } // 规避 GEMV 模式

        // cmatrixInitVal：仅第一块（k==0）时初始化 L0C，其余块累加
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource  = false;

        // unitFlag：中间轮设为 ENABLE（2），最后一轮（若外层也是最后）设为 EN_OUTER_LAST（3）
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1)
                ? UNITFLAG_EN_OUTER_LAST
                : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        // 释放 L0A、L0B，允许 MTE1 加载下一块
        l0aBuffer.Set<HardEvent::M_MTE1>();
        l0bBuffer.Set<HardEvent::M_MTE1>();
    }
}

// ============================================================
// MatmulMMx：沿 M 轴分块的 MX 量化矩阵乘（310/310R6/5102 架构）
// ============================================================
/**
 * @brief 将 M 维度切分为多个 baseM 大小的块，B 整体载入 L0B，A 逐块载入 L0A
 *
 * 适用于 M 维度较大的情况。
 * 策略：B 只需加载一次（整体放入 L0B），A 分块循环加载。
 * 对应计算：
 *   LoadData B → L0B（一次）
 *   for m in range(0, M, baseM):
 *       LoadData A[m:m+baseM, :] → L0A
 *       Mmad → C[m:m+baseM, :]
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType,
          typename AScaleType = float, typename BScaleType = float,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulMMx(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<AScaleType>& aScaleL1Tensor,
    const LocalTensor<BScaleType>& bScaleL1Tensor,
    L0AType& aL0BuffsDb,
    L0BType& bL0BuffsDb,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    // 计算 M 方向的循环次数
    uint32_t mLoops   = (param.singleM + baseM - 1) / baseM;
    uint32_t tailSize = param.singleM % baseM;
    uint32_t tailM    = tailSize ? tailSize : baseM;

    // A 在 L1 中每块（baseM 行）的地址步长
    // MK 布局（非转置）：步长 = K 对齐后 × baseM 个元素
    // KM 布局（已转置）：步长 = baseM × 16（分形维度）
    uint64_t L1Aoffset = param.isLeftTranspose
        ? baseM << 4
        : ((param.singleK + 15) >> 4 << 4) * baseM;

#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value ||
                  IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value) {
        L1Aoffset = ((param.singleK + 31) >> 5 << 5) * baseM;
    }
#endif

    // L0C 中每个 M 分块的地址步长（C 矩阵按 M×N 展开，N 方向 32 对齐）
    uint64_t L0Coffset = ((param.singleN + 31) >> 5 << 5) * baseM;

    // 步骤1：B 整体加载到 L0B（只做一次）
    Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.GetTensor<L0BDType>();

    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor,
                                     param, 0, param.singleK, param.singleN);
    } else if constexpr (IsSameType<L0BDType, fp8_e4m3fn_t>::value) {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0bBuffer.Set<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>(); // 确保 B 已完全加载完毕

    // 步骤2：逐块加载 A 并计算
    for (uint32_t m = 0; m < mLoops; m++) {
        uint32_t tileM = (m == (mLoops - 1)) ? tailM : baseM;

        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.GetTensor<L0ADType>();

        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType>(L0ATensor, aL1Tensor, aScaleL1Tensor,
                                         param, m * L1Aoffset, tileM, param.singleK);
        }
        l0aBuffer.Set<HardEvent::MTE1_M>();
        l0aBuffer.Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = tileM;
        mmadParams.n = param.singleN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) { mmadParams.m = 16; }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (m == 0); // 仅第一块清零 L0C
        mmadParams.cmatrixSource  = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (m == mLoops - 1)
                ? UNITFLAG_EN_OUTER_LAST
                : UNITFLAG_ENABLE;
        }

        // 注意：C 的输出写到对应 M 块的偏移位置
        Mmad(cL0Tensor[m * L0Coffset], L0ATensor, L0BTensor, mmadParams);
        l0aBuffer.Set<HardEvent::M_MTE1>();
    }

    // 释放 L0B
    l0bBuffer.Set<HardEvent::M_MTE1>();
}

// ============================================================
// 以下为非 310/310R6/5102 架构（如昇腾 910 系列）的实现
// ============================================================
#else

// 是否重置 Load3D 配置（isSetFMatrix=true, isSetPadding=true）
static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};

/**
 * @brief 910 系列架构：A 矩阵从 L1 加载到 L0A（使用 Load3D 指令）
 *
 * 910 系列的 AI Core 使用 Load3D（三维卷积式加载）指令代替 LoadData2D，
 * 通过将矩阵乘法映射为等价的 1×1 卷积操作来完成分形格式转换。
 *
 * 参数映射（将矩阵维度映射为卷积参数）：
 *   - l1H = mSplitSize / 16  → 卷积的"图像高度"（每 16 行为一个分形）
 *   - l1W = 16               → 卷积的"图像宽度"（固定为 16，对应分形 width）
 *   - mExtension = mSplitSize → 目标 M 方向长度
 *   - kExtension = kSplitSize → 目标 K 方向长度
 *   - strideW/H = 1          → 滑动步长为 1（等价于普通加载，无跳步）
 *   - filterW/H = 1          → 卷积核大小为 1×1
 *   - dilationFilterW/H = 1  → 无膨胀
 *
 * @tparam T   数据类型
 * @tparam AL  布局（MK 或 KM）
 */
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(
    LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
    const MMParam& mmParam, uint64_t L1Aoffset,
    uint32_t kSplitSize, uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        // MK 布局：A 以行主序存储，使用 Load3D 的标准 MK→L0A 路径
        LoadData3DParamsV2<T> loadData3DParams;

        loadData3DParams.l1H = mSplitSize / LOAD3D_L1W_SIZE; // 源矩阵的"图像高度"（M÷16）
        loadData3DParams.l1W = LOAD3D_L1W_SIZE;               // 源矩阵的"图像宽度"（固定16）

        // padding 设置：前3个方向为0，尾部padding=255（尾部数据不影响滑窗结果）
        loadData3DParams.padList[0] = 0;   // 上padding
        loadData3DParams.padList[1] = 0;   // 下padding
        loadData3DParams.padList[2] = 0;   // 左padding
        loadData3DParams.padList[3] = 255; // 右padding（尾部数据标记）

        loadData3DParams.mExtension = mSplitSize; // 目标 M 方向传输长度
        loadData3DParams.kExtension = kSplitSize; // 目标 K 方向传输长度
        loadData3DParams.mStartPt   = 0;          // 卷积核在目标 width 维度的起点
        loadData3DParams.kStartPt   = 0;          // 卷积核在目标 height 维度的起点

        // 卷积滑动参数（1×1 卷积，步长=1，无膨胀 → 等价于直接复制）
        loadData3DParams.strideW        = LOAD3D_STRIDE_W;      // W 方向步长 = 1
        loadData3DParams.strideH        = LOAD3D_STRIDE_H;      // H 方向步长 = 1
        loadData3DParams.filterW        = LOAD3D_FILTER_W;      // 卷积核 width = 1
        loadData3DParams.filterSizeW    = false;                  // 不扩展 filterW
        loadData3DParams.filterH        = LOAD3D_FILTER_H;      // 卷积核 height = 1
        loadData3DParams.filterSizeH    = false;                  // 不扩展 filterH
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W; // 膨胀系数 = 1
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H;

        loadData3DParams.enTranspose  = 0;   // 不启用矩阵级转置（分形内部转置由硬件处理）
        loadData3DParams.fMatrixCtrl  = 0;   // 不使用 F 矩阵控制

        // channelSize：源操作数的通道数，等于 K 轴大小
        // 膨胀系数为1时，目的width = channelSize = K
        loadData3DParams.channelSize  = kSplitSize;

        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams, LOAD3DV2_CONFIG);

    } else {
        // KM 布局（A 已转置，以列主序存储）：
        // 使用 Load3D 的转置路径，enTranspose=1
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H         = kSplitSize / LOAD3D_L1W_SIZE;
        loadData3DParams.l1W         = LOAD3D_L1W_SIZE;
        loadData3DParams.padList[0]  = 0;
        loadData3DParams.padList[1]  = 0;
        loadData3DParams.padList[2]  = 0;
        loadData3DParams.padList[3]  = 255;
        loadData3DParams.mExtension  = kSplitSize; // 注意：转置后 M 和 K 的语义互换
        loadData3DParams.kExtension  = mSplitSize;
        loadData3DParams.mStartPt    = 0;
        loadData3DParams.kStartPt    = 0;
        loadData3DParams.strideW         = LOAD3D_STRIDE_W;
        loadData3DParams.strideH         = LOAD3D_STRIDE_H;
        loadData3DParams.filterW         = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW     = false;
        loadData3DParams.filterH         = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH     = false;
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W;
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H;
        loadData3DParams.enTranspose     = 1;  // 启用转置
        loadData3DParams.fMatrixCtrl     = 0;
        loadData3DParams.channelSize     = mSplitSize;

        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams, LOAD3DV2_CONFIG);
    }
}

/**
 * @brief 910 系列架构：B 矩阵从 L1 加载到 L0B（使用 Load3D 指令）
 *
 * 与 A 矩阵加载对称。KN 布局（非转置）时使用标准路径，
 * NK 布局（已转置）时使用 enTranspose=1 路径。
 *
 * @tparam T   数据类型
 * @tparam BL  布局（KN 或 NK）
 */
template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(
    LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
    const MMParam& mmParam, uint64_t L1Boffset,
    uint32_t kSplitSize, uint32_t nSplitSize)
{
    if constexpr (BL == ABLayout::KN) {
        // KN 布局：B 以行主序存储（K 行 N 列）
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H         = kSplitSize / LOAD3D_L1W_SIZE;
        loadData3DParams.l1W         = LOAD3D_L1W_SIZE;
        loadData3DParams.padList[0]  = 0;
        loadData3DParams.padList[1]  = 0;
        loadData3DParams.padList[2]  = 0;
        loadData3DParams.padList[3]  = 255;
        loadData3DParams.mExtension  = nSplitSize; // Load3D 中 B 的 M 对应 N 维度
        loadData3DParams.kExtension  = kSplitSize;
        loadData3DParams.mStartPt    = 0;
        loadData3DParams.kStartPt    = 0;
        loadData3DParams.strideW         = LOAD3D_STRIDE_W;
        loadData3DParams.strideH         = LOAD3D_STRIDE_H;
        loadData3DParams.filterW         = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW     = false;
        loadData3DParams.filterH         = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH     = false;
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W;
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H;
        loadData3DParams.enTranspose     = 1;  // KN→L0B 需要转置加载
        loadData3DParams.fMatrixCtrl     = 0;
        loadData3DParams.channelSize     = nSplitSize;

        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams, LOAD3DV2_CONFIG);
    } else {
        // NK 布局：B 以列主序存储（N 行 K 列，即 B^T）
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H         = nSplitSize / LOAD3D_L1W_SIZE;
        loadData3DParams.l1W         = LOAD3D_L1W_SIZE;
        loadData3DParams.padList[0]  = 0;
        loadData3DParams.padList[1]  = 0;
        loadData3DParams.padList[2]  = 0;
        loadData3DParams.padList[3]  = 255;
        loadData3DParams.mExtension  = nSplitSize;
        loadData3DParams.kExtension  = kSplitSize;
        loadData3DParams.mStartPt    = 0;
        loadData3DParams.kStartPt    = 0;
        loadData3DParams.strideW         = LOAD3D_STRIDE_W;
        loadData3DParams.strideH         = LOAD3D_STRIDE_H;
        loadData3DParams.filterW         = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW     = false;
        loadData3DParams.filterH         = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH     = false;
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W;
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H;
        loadData3DParams.enTranspose     = 0;  // NK 直接加载（无需额外转置）
        loadData3DParams.fMatrixCtrl     = 0;
        loadData3DParams.channelSize     = kSplitSize;

        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams, LOAD3DV2_CONFIG);
    }
}

#endif  // 结束 310/310R6/5102 vs 其他架构的条件编译

// ============================================================
// ============================================================
// 以下为对外暴露的主要 Matmul 接口（通用于所有架构）
// 根据分块策略分为以下几种：
//   MatmulFull  — 全载（不分块）
//   MatmulK     — 沿 K 轴分块
//   MatmulN     — 沿 N 轴分块
//   MatmulKM    — 沿 K 和 M 双轴分块
//   MatmulKPP   — K 轴分块 + 流水线并行
// ============================================================
// ============================================================

// ============================================================
// MatmulFull：全载矩阵乘（不分块）
// ============================================================
/**
 * @brief 最简单的矩阵乘：A 和 B 整体装载到 L0，执行一次 Mmad
 *
 * 适用场景：
 *   - 矩阵 M、K、N 均较小，能够同时放入 L0A 和 L0B
 *   - 每个 AI Core 只负责一个矩阵块的计算（不需要外层分块循环）
 *
 * 调用示例（典型用法）：
 * @code
 *   MMParam param = MakeMMParam(32, 32, 32, false, false);
 *   MatmulFull<half, half, float,
 *              16, 16, 16,
 *              ABLayout::MK, ABLayout::KN>(
 *       aL1, bL1, cL0, param);
 * @endcode
 *
 * @tparam A     A 矩阵元素类型（如 half、float、fp8_e4m3fn_t）
 * @tparam B     B 矩阵元素类型
 * @tparam C     C 矩阵元素类型（输出，通常为 float 或 int32）
 * @tparam baseM M 方向分形基本单位（必须为 16 的整数倍）
 * @tparam baseN N 方向分形基本单位（必须为 16 的整数倍）
 * @tparam baseK K 方向分形基本单位（必须为 16 的整数倍）
 * @tparam AL    A 矩阵在 L1 中的布局（MK=行主序，KM=列主序）
 * @tparam BL    B 矩阵在 L1 中的布局（KN=行主序，NK=列主序）
 * @param aL1Tensor  L1 中的 A 矩阵数据
 * @param bL1Tensor  L1 中的 B 矩阵数据
 * @param cL0Tensor  L0C 中的输出 C 矩阵（Mmad 直接写入此处）
 * @param param      矩阵乘法参数（singleM/N/K、是否转置等）
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulFull(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    // ---- 加载 A 到 L0A ----
    // 在硬件上，L0A 是 ping-pong 双缓冲（两块交替使用），
    // 这里直接声明一个简单的单缓冲 LocalTensor 用于全载场景
    LocalTensor<A> aL0Tensor;  // L0A 张量（由调用者提前通过 pipe 分配）

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    // 310/310R6/5102：使用 LoadData2D 指令
    LoadDataToL0A<A>(aL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
#else
    // 910 等：使用 Load3D 指令
    LoadDataToL0A<A, AL>(aL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
#endif

    // ---- 加载 B 到 L0B ----
    LocalTensor<B> bL0Tensor;

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    LoadDataToL0B<B>(bL0Tensor, bL1Tensor, param, 0, param.singleK, param.singleN);
#else
    LoadDataToL0B<B, BL>(bL0Tensor, bL1Tensor, param, 0, param.singleK, param.singleN);
#endif

    // ---- 执行 Mmad ----
    MmadParams mmadParams;
    mmadParams.m             = param.singleM;
    if (param.realM != 0) { mmadParams.m = param.realM; }
    mmadParams.n             = param.singleN;
    mmadParams.k             = param.singleK;
    mmadParams.cmatrixInitVal = param.cmatrixInitVal; // 是否清零 L0C
    mmadParams.cmatrixSource  = false;
    mmadParams.unitFlag       = param.unitFlag;
    if (mmadParams.m == 1) { mmadParams.m = 16; }     // 规避 GEMV 模式

    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
}

// ============================================================
// MatmulK：沿 K 轴分块的矩阵乘
// ============================================================
/**
 * @brief 将 K 维度切分为多个 baseK 大小的块，逐块累加到 L0C
 *
 * 适用场景：K 维度较大（如 Transformer 中的 head_dim），无法整体放入 L0A
 *
 * 分块策略（以 K=96, baseK=32 为例）：
 *   轮0：A[:,  0:32] × B[ 0:32, :] → L0C（初始化）
 *   轮1：A[:, 32:64] × B[32:64, :] → L0C（累加）
 *   轮2：A[:, 64:96] × B[64:96, :] → L0C（累加）
 *
 * 与 MatmulKMx 的区别：MatmulK 用于普通数据类型（half/float），
 * MatmulKMx 用于 MX 量化数据类型。
 *
 * @param param.unitFlag  外层切 K 场景中，外层最后一轮需传入 UNITFLAG_EN_OUTER_LAST（3）
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulK(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    uint32_t kLoops   = (param.singleK + baseK - 1) / baseK; // K 方向块数
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK    = tailSize ? tailSize : baseK;           // 尾块大小

    // A 在 L1 中每块的地址步长（元素数）
    // MK 布局：连续的 K 轴，步长 = M对齐行数 × baseK
    // KM 布局：转置存储，步长 = baseK × 16
    uint64_t L1Aoffset = (AL == ABLayout::KM)
        ? (uint64_t)baseK << 4
        : ((uint64_t)((param.singleM + 15) >> 4 << 4)) * baseK;

    // B 在 L1 中每块的地址步长
    // KN 布局：步长 = baseK × 16（K 轴连续）
    // NK 布局：转置存储，步长 = N对齐行数 × baseK
    uint64_t L1Boffset = (BL == ABLayout::NK)
        ? ((uint64_t)((param.singleN + 15) >> 4 << 4)) * baseK
        : (uint64_t)baseK << 4;

    // K 分块主循环
    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;

        // 声明本地缓冲区（由 pipe 管理，此处仅声明引用）
        LocalTensor<A> aL0Tensor;
        LocalTensor<B> bL0Tensor;

        // 加载 A 的第 k 块
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
        LoadDataToL0A<A>(aL0Tensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
#else
        LoadDataToL0A<A, AL>(aL0Tensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
#endif

        // 加载 B 的第 k 块（NK 布局时 B 无需沿 K 移动，loopNum=1）
        uint64_t bOffset = (BL == ABLayout::NK) ? 0 : k * L1Boffset;
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
        LoadDataToL0B<B>(bL0Tensor, bL1Tensor, param, bOffset, tileK, param.singleN);
#else
        LoadDataToL0B<B, BL>(bL0Tensor, bL1Tensor, param, bOffset, tileK, param.singleN);
#endif

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) { mmadParams.m = param.realM; }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) { mmadParams.m = 16; }

        // 仅第一块初始化 L0C，其余块累加
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource  = false;

        // unitFlag：中间轮=ENABLE(2)，最后一轮（且外层也是最后）=EN_OUTER_LAST(3)
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1)
                ? UNITFLAG_EN_OUTER_LAST
                : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
    }
}

// ============================================================
// MatmulN：沿 N 轴分块的矩阵乘
// ============================================================
/**
 * @brief 将 N 维度切分为多个 baseN 大小的块，A 整体载入，B 逐块载入
 *
 * 适用场景：N 维度较大（如输出 Feature Map 的宽度），无法整体放入 L0B
 *
 * 分块策略：
 *   A 整体加载到 L0A（一次）
 *   for n in range(0, N, baseN):
 *       B[:, n:n+baseN] → L0B
 *       Mmad → C[:, n:n+baseN]
 *
 * C 矩阵的 N 方向分块结果写到 L0C 的不同偏移位置。
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulN(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    uint32_t nLoops   = (param.singleN + baseN - 1) / baseN;
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN    = tailSize ? tailSize : baseN;

    // B 在 L1 中每块的 N 方向步长
    uint64_t L1Boffset = (BL == ABLayout::NK)
        ? (uint64_t)baseN * ((param.singleK + 15) >> 4 << 4)  // NK 布局
        : (uint64_t)baseN << 4;                                 // KN 布局

    // L0C 中每个 N 分块的偏移（C 矩阵 M 行，N 方向依次排列）
    uint64_t L0Coffset = ((uint64_t)((param.singleM + 31) >> 5 << 5)) * baseN;

    // A 整体加载一次
    LocalTensor<A> aL0Tensor;
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    LoadDataToL0A<A>(aL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
#else
    LoadDataToL0A<A, AL>(aL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
#endif

    // N 分块循环
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        LocalTensor<B> bL0Tensor;
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
        LoadDataToL0B<B>(bL0Tensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN);
#else
        LoadDataToL0B<B, BL>(bL0Tensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN);
#endif

        // 构造本轮的 MMParam（N 大小变为 tileN）
        MMParam tileParam  = param;
        tileParam.singleN  = tileN;

        MmadParams mmadParams;
        mmadParams.m             = param.singleM;
        if (param.realM != 0)  { mmadParams.m = param.realM; }
        mmadParams.n             = tileN;
        mmadParams.k             = param.singleK;
        if (mmadParams.m == 1) { mmadParams.m = 16; }
        mmadParams.cmatrixInitVal = param.cmatrixInitVal;
        mmadParams.cmatrixSource  = false;
        mmadParams.unitFlag       = param.unitFlag;

        // C 的结果写到对应 N 块的偏移位置
        Mmad(cL0Tensor[n * L0Coffset], aL0Tensor, bL0Tensor, mmadParams);
    }
}

// ============================================================
// MatmulKM：沿 K 和 M 双轴分块的矩阵乘
// ============================================================
/**
 * @brief 同时在 K 和 M 方向分块
 *
 * 适用场景：K 和 M 均很大，L0A 甚至无法放下单行 A（M 方向需要分块）
 *
 * 分块策略（外层 K，内层 M）：
 *   for k in range(0, K, baseK):
 *       B[k:k+baseK, :] → L0B（每轮 K 加载一次）
 *       for m in range(0, M, baseM):
 *           A[m:m+baseM, k:k+baseK] → L0A
 *           Mmad → C[m:m+baseM, :]
 *
 * L0C 按 M 分块写入不同偏移（C[m*baseM : (m+1)*baseM, :]）。
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKM(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    uint32_t kLoops   = (param.singleK + baseK - 1) / baseK;
    uint32_t tailK    = (param.singleK % baseK) ? (param.singleK % baseK) : baseK;

    uint32_t mLoops   = (param.singleM + baseM - 1) / baseM;
    uint32_t tailM    = (param.singleM % baseM) ? (param.singleM % baseM) : baseM;

    // A：K 方向步长（同 MatmulK）
    uint64_t L1AKoffset = (AL == ABLayout::KM)
        ? (uint64_t)baseK << 4
        : ((uint64_t)((param.singleM + 15) >> 4 << 4)) * baseK;

    // A：M 方向步长
    uint64_t L1AMoffset = (AL == ABLayout::KM)
        ? (uint64_t)baseM * ((param.singleK + 15) >> 4 << 4)
        : (uint64_t)baseM << 4;

    // B：K 方向步长（同 MatmulK）
    uint64_t L1Boffset  = (BL == ABLayout::NK)
        ? ((uint64_t)((param.singleN + 15) >> 4 << 4)) * baseK
        : (uint64_t)baseK << 4;

    // L0C：M 分块偏移
    uint64_t L0Coffset  = ((uint64_t)((param.singleN + 31) >> 5 << 5)) * baseM;

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;

        // 每轮 K 加载 B 的对应 K 块
        LocalTensor<B> bL0Tensor;
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
        LoadDataToL0B<B>(bL0Tensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN);
#else
        LoadDataToL0B<B, BL>(bL0Tensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN);
#endif

        for (uint32_t m = 0; m < mLoops; m++) {
            uint32_t tileM = (m == (mLoops - 1)) ? tailM : baseM;

            // A[m,k] 块的地址 = M方向偏移 + K方向偏移
            uint64_t aOffset = m * L1AMoffset + k * L1AKoffset;

            LocalTensor<A> aL0Tensor;
            // 构造当前块的 MMParam（M 大小变为 tileM）
            MMParam tileParam  = param;
            tileParam.singleM  = tileM;

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
            LoadDataToL0A<A>(aL0Tensor, aL1Tensor, tileParam, aOffset, tileK, tileM);
#else
            LoadDataToL0A<A, AL>(aL0Tensor, aL1Tensor, tileParam, aOffset, tileK, tileM);
#endif

            MmadParams mmadParams;
            mmadParams.m = tileM;
            mmadParams.n = param.singleN;
            mmadParams.k = tileK;
            if (mmadParams.m == 1) { mmadParams.m = 16; }
            // 仅第一个 K 块且第一个 M 块时初始化 L0C 对应区域
            mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
            mmadParams.cmatrixSource  = false;

            if (param.unitFlag != 0) {
                bool isLast = (param.unitFlag == UNITFLAG_EN_OUTER_LAST)
                              && (k == kLoops - 1) && (m == mLoops - 1);
                mmadParams.unitFlag = isLast ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
            }

            // 写到 C 矩阵对应 M 块的偏移位置
            Mmad(cL0Tensor[m * L0Coffset], aL0Tensor, bL0Tensor, mmadParams);
        }
    }
}

// ============================================================
// MatmulKPP：K 轴分块 + 流水线并行（Ping-Pong Pipeline）
// ============================================================
/**
 * @brief K 分块矩阵乘的流水线优化版本
 *
 * 在 MatmulK 的基础上，利用 L0A/L0B 双缓冲（ping-pong buffer）
 * 实现数据搬运（MTE1）和矩阵计算（Mmad/M单元）的重叠，
 * 最大化 AI Core 的硬件利用率。
 *
 * 流水线示意（以 3 块 K 为例）：
 *   时间 →
 *   MTE1:  [加载K0]  [加载K1]  [加载K2]
 *   M单元:           [Mmad K0] [Mmad K1] [Mmad K2]
 *
 * 硬件事件同步：
 *   HardEvent::M_MTE1   → Mmad 通知 MTE1 可以开始加载下一块
 *   HardEvent::MTE1_M   → MTE1 通知 Mmad 数据已就绪
 *
 * 相比 MatmulK，MatmulKPP 需要 L0AType/L0BType 支持双缓冲接口（.Get()/.Set()）
 *
 * @tparam L0AType  L0A 双缓冲对象类型（如 TBuf<TPosition::A1> 的双缓冲包装）
 * @tparam L0BType  L0B 双缓冲对象类型
 */
template <typename A, typename B, typename C,
          uint32_t baseM, uint32_t baseN, uint32_t baseK,
          ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulKPP(
    const LocalTensor<A>& aL1Tensor,
    const LocalTensor<B>& bL1Tensor,
    L0AType& aL0BuffsDb,
    L0BType& bL0BuffsDb,
    const LocalTensor<C>& cL0Tensor,
    const MMParam& param)
{
    uint32_t kLoops   = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK    = tailSize ? tailSize : baseK;

    uint64_t L1Aoffset = (AL == ABLayout::KM)
        ? (uint64_t)baseK << 4
        : ((uint64_t)((param.singleM + 15) >> 4 << 4)) * baseK;

    uint64_t L1Boffset = (BL == ABLayout::NK)
        ? ((uint64_t)((param.singleN + 15) >> 4 << 4)) * baseK
        : (uint64_t)baseK << 4;

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;

        // ---- 流水线：从双缓冲池获取 L0A/L0B，等待上一轮 Mmad 完成 ----
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // MTE1 等 Mmad：上轮计算完才能写入新数据

        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>();

        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();

        // ---- 加载当前 K 块 ----
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
        LoadDataToL0A<A>(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        uint64_t bOffset = (BL == ABLayout::NK) ? 0 : k * L1Boffset;
        LoadDataToL0B<B>(L0BTensor, bL1Tensor, param, bOffset, tileK, param.singleN);
#else
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        uint64_t bOffset = (BL == ABLayout::NK) ? 0 : k * L1Boffset;
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, bOffset, tileK, param.singleN);
#endif

        // ---- 通知 Mmad：数据就绪 ----
        l0aBuffer.Set<HardEvent::MTE1_M>();
        l0bBuffer.Set<HardEvent::MTE1_M>();

        // ---- 等待本轮数据就绪，执行 Mmad ----
        l0aBuffer.Wait<HardEvent::MTE1_M>();
        l0bBuffer.Wait<HardEvent::MTE1_M>();

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) { mmadParams.m = param.realM; }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) { mmadParams.m = 16; }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource  = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1)
                ? UNITFLAG_EN_OUTER_LAST
                : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        // ---- 释放双缓冲，允许下一轮 MTE1 提前加载 ----
        l0aBuffer.Set<HardEvent::M_MTE1>();
        l0bBuffer.Set<HardEvent::M_MTE1>();
    }
}

} // namespace fa_base_matmul

#endif // MATMUL_H

