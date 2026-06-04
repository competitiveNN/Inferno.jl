#ifdef __AVX2__
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// Convert F32 vector to BF16 using truncation
void fp32_to_bf16_row(const float *src, uint16_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } u;
        u.f = src[i];
        dst[i] = (uint16_t)(u.u >> 16);
    }
}

// Simple BF16 GEMV: out[i] = sum_j A[i,j] * x[j]
// A is (m x n) in row-major, x is n elements
void bf16_gemv_avx2(const uint16_t *A, const uint16_t *x, float *out,
                     int m, int n, int lda) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        const uint16_t *row = A + i * lda;
        for (int j = 0; j < n; j++) {
            // Convert BF16 to float
            uint32_t u = ((uint32_t)row[j]) << 16;
            float a;
            *(uint32_t*)&a = u;
            uint32_t xv = ((uint32_t)x[j]) << 16;
            float xv_f;
            *(uint32_t*)&xv_f = xv;
            sum += a * xv_f;
        }
        out[i] = sum;
    }
}

// Compute out[k] = dot(A[:, col_start+k], x) for k in [0, n_cols)
void bf16_gemv_cols_avx2(const uint16_t *A, const uint16_t *x, float *out,
                          int col_start, int n_cols, int col_len) {
    for (int k = 0; k < n_cols; k++) {
        float sum = 0.0f;
        const uint16_t *col = A + (col_start + k) * col_len;
        for (int j = 0; j < col_len; j++) {
            uint32_t u = ((uint32_t)col[j]) << 16;
            float a;
            *(uint32_t*)&a = u;
            uint32_t xv = ((uint32_t)x[j]) << 16;
            float xv_f;
            *(uint32_t*)&xv_f = xv;
            sum += a * xv_f;
        }
        out[k] = sum;
    }
}
#else
// Fallback for non-AVX2 platforms
#include <stdint.h>
#include <stddef.h>

void fp32_to_bf16_row(const float *src, uint16_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } u;
        u.f = src[i];
        dst[i] = (uint16_t)(u.u >> 16);
    }
}

void bf16_gemv_avx2(const uint16_t *A, const uint16_t *x, float *out,
                     int m, int n, int lda) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        const uint16_t *row = A + i * lda;
        for (int j = 0; j < n; j++) {
            uint32_t u = ((uint32_t)row[j]) << 16;
            float a;
            *(uint32_t*)&a = u;
            uint32_t xv = ((uint32_t)x[j]) << 16;
            float xv_f;
            *(uint32_t*)&xv_f = xv;
            sum += a * xv_f;
        }
        out[i] = sum;
    }
}

void bf16_gemv_cols_avx2(const uint16_t *A, const uint16_t *x, float *out,
                          int col_start, int n_cols, int col_len) {
    for (int k = 0; k < n_cols; k++) {
        float sum = 0.0f;
        const uint16_t *col = A + (col_start + k) * col_len;
        for (int j = 0; j < col_len; j++) {
            uint32_t u = ((uint32_t)col[j]) << 16;
            float a;
            *(uint32_t*)&a = u;
            uint32_t xv = ((uint32_t)x[j]) << 16;
            float xv_f;
            *(uint32_t*)&xv_f = xv;
            sum += a * xv_f;
        }
        out[k] = sum;
    }
}
#endif
