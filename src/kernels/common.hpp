#pragma once

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes
// cg(cache global, L2): only support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer)                                            \
  (reinterpret_cast<const float4 *>(&(pointer))[0])

#define MAC_8(C, A, B, a_comp, b_comp)                                         \
  C += A[0].a_comp * B[0].b_comp;                                              \
  C += A[1].a_comp * B[1].b_comp;                                              \
  C += A[2].a_comp * B[2].b_comp;                                              \
  C += A[3].a_comp * B[3].b_comp;                                              \
  C += A[4].a_comp * B[4].b_comp;                                              \
  C += A[5].a_comp * B[5].b_comp;                                              \
  C += A[6].a_comp * B[6].b_comp;                                              \
  C += A[7].a_comp * B[7].b_comp;

#define COMPUTE_1X8(C_ROW, A, B0, B1, a_comp)                                  \
  MAC_8(C_ROW[0], A, B0, a_comp, x);                                           \
  MAC_8(C_ROW[1], A, B0, a_comp, y);                                           \
  MAC_8(C_ROW[2], A, B0, a_comp, z);                                           \
  MAC_8(C_ROW[3], A, B0, a_comp, w);                                           \
                                                                               \
  MAC_8(C_ROW[4], A, B1, a_comp, x);                                           \
  MAC_8(C_ROW[5], A, B1, a_comp, y);                                           \
  MAC_8(C_ROW[6], A, B1, a_comp, z);                                           \
  MAC_8(C_ROW[7], A, B1, a_comp, w);

#define MAC_4(C, A, B, a_comp, b_comp)                                         \
  C += A[0].a_comp * B[0].b_comp;                                              \
  C += A[1].a_comp * B[1].b_comp;                                              \
  C += A[2].a_comp * B[2].b_comp;                                              \
  C += A[3].a_comp * B[3].b_comp;

#define COMPUTE_1X8_K4(C_ROW, A, B0, B1, a_comp)                               \
  MAC_4(C_ROW[0], A, B0, a_comp, x);                                           \
  MAC_4(C_ROW[1], A, B0, a_comp, y);                                           \
  MAC_4(C_ROW[2], A, B0, a_comp, z);                                           \
  MAC_4(C_ROW[3], A, B0, a_comp, w);                                           \
                                                                               \
  MAC_4(C_ROW[4], A, B1, a_comp, x);                                           \
  MAC_4(C_ROW[5], A, B1, a_comp, y);                                           \
  MAC_4(C_ROW[6], A, B1, a_comp, z);                                           \
  MAC_4(C_ROW[7], A, B1, a_comp, w);

#define MAC_2(C, A, B, a_comp, b_comp)                                         \
  C += A[0].a_comp * B[0].b_comp;                                              \
  C += A[1].a_comp * B[1].b_comp;

#define COMPUTE_1X8_K2(C_ROW, A, B0, B1, a_comp)                               \
  MAC_2(C_ROW[0], A, B0, a_comp, x);                                           \
  MAC_2(C_ROW[1], A, B0, a_comp, y);                                           \
  MAC_2(C_ROW[2], A, B0, a_comp, z);                                           \
  MAC_2(C_ROW[3], A, B0, a_comp, w);                                           \
                                                                               \
  MAC_2(C_ROW[4], A, B1, a_comp, x);                                           \
  MAC_2(C_ROW[5], A, B1, a_comp, y);                                           \
  MAC_2(C_ROW[6], A, B1, a_comp, z);                                           \
  MAC_2(C_ROW[7], A, B1, a_comp, w);

template <int A_N, int B_N> struct Stage {
  float A[A_N];
  float B[B_N];
};
