/*
 * SIMD dequantize+mul kernels for K-quants (Q4_K, Q5_K, Q6_K, Q8_0).
 * Faithful port of llama.cpp's ggml-cpu/arch/x86/quants.c
 *
 * Key optimization: quantize F32 input to Q8_K (int8) ONCE per token,
 * then do integer dot products using _mm256_maddubs_epi16 which
 * does 32 unsigned×signed byte multiplies and 16 adds in ONE instruction.
 */

#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

#define QK_K 256
#define QK8_0 32
#define K_SCALE_SIZE 12
#define GGML_RESTRICT restrict

/* ---- Block structures ---- */

// Q4_K block: 144 bytes for 256 values
typedef struct {
    uint16_t d;              // super-block scale (Float16)
    uint16_t dmin;           // super-block min scale (Float16)
    uint8_t  scales[K_SCALE_SIZE]; // scales and mins, packed 6-bit
    uint8_t  qs[QK_K/2];    // 4-bit quants (2 per byte)
} block_q4_K;

// Q5_K block: 176 bytes for 256 values
typedef struct {
    uint16_t d;              // super-block scale (Float16)
    uint16_t dmin;           // super-block min scale (Float16)
    uint8_t  scales[K_SCALE_SIZE]; // scales and mins, packed 6-bit
    uint8_t  qh[QK_K/8];    // high bits (1 per value)
    uint8_t  qs[QK_K/2];    // low 4 bits (2 per byte)
} block_q5_K;

// Q6_K block: 210 bytes for 256 values
// 6-bit signed quantization (range -32 to 31)
typedef struct {
	uint8_t ql[QK_K/2];     // quants, lower 4 bits
	uint8_t qh[QK_K/4];     // quants, upper 2 bits
	int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
	uint16_t d;              // super-block scale (Float16)
} block_q6_K;

// Q8_K block: intermediate quantization for dot products
// sizeof = 4 + 256 + 32 = 292 bytes, but may be padded
typedef struct {
	float d;            // delta (scale)
	int8_t qs[QK_K];    // quants
	int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

// Q8_0 block: 34 bytes for 32 values
typedef struct {
    uint16_t d;              // scale (Float16)
    int8_t   qs[QK8_0];     // quants
} block_q8_0;

/* ---- Helpers ---- */

// FP16 -> FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exponent = (h >> 10) & 0x1f;
    uint32_t mantissa = h & 0x3ff;
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) { f = sign << 31; }
        else {
            exponent = 127 - 15 + 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa = (mantissa & 0x3ff) << 13;
            f = (sign << 31) | (exponent << 23) | mantissa;
        }
    } else if (exponent == 0x1f) {
        f = (sign << 31) | 0x7f800000 | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Horizontal sum of 8 floats in __m256
static inline float hsum_float_8(const __m256 x) {
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

// MM256_SET_M128I: combine two 128-bit vectors into 256-bit
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

/*
 * Scale shuffle table for K-quants (from llama.cpp).
 * After cvtepu8_epi16, each scale is 2 bytes, so pairs of byte indices.
 * 8 entries * 32 bytes each = 256 byte table.
 */
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
       10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
       12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
       14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
	return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

/*
 * Scale shuffle table for Q6_K (16 int8 scales, 128-bit shuffle).
 * Each scale byte is broadcast to 8 positions within a 128-bit lane.
 * 8 entries * 16 bytes each = 128 byte table.
 */
static inline __m128i get_scale_shuffle(int i) {
	static const uint8_t k_shuffle[128] = {
		0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
		4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
		6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
		8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
		10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
		12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
		14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
	};
	return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}

/*
 * Integer multiply-sum of signed i8 pairs -> float.
 * Uses sign manipulation to use _mm256_maddubs_epi16 (unsigned×signed)
 * for signed×signed multiplication.
 */
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    // Get absolute values of x
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of y with the signs of x
    const __m256i sy = _mm256_sign_epi8(y, x);
    // Now we have unsigned×signed = effectively signed×signed
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_maddubs_epi16(ax, sy);
    // Convert to 32-bit and accumulate
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed = _mm256_madd_epi16(summed_pairs, ones);
    return _mm256_cvtepi32_ps(summed);
}

/* ---- Quantization ---- */

/*
 * Quantize a row of F32 values to Q8_K blocks.
 * Called ONCE per token for the activation vector.
 */
void quantize_row_q8_K(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = (int)(k / QK_K);

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        const float * src = x + i * QK_K;
        for (int j = 0; j < QK_K; j++) {
            float ax = fabsf(src[j]);
            if (ax > amax) amax = ax;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f/d : 0.0f;
        y[i].d = d;

        int8_t * dst = y[i].qs;
        int16_t * bsums = y[i].bsums;

        for (int j = 0; j < QK_K; j += 16) {
            int sum = 0;
            for (int l = 0; l < 16; l++) {
                int8_t q = (int8_t)roundf(src[j+l] * id);
                dst[j+l] = q;
                sum += (int)q;
            }
            bsums[j/16] = (int16_t)sum;
        }
    }
}

/* ---- Dot products ---- */

/*
 * Q4_K × Q8_K dot product (the fast path from llama.cpp)
 * Computes: sum_i (dequant_q4_K(x[i]) * dequant_q8_K(y[i]))
 * Uses integer SIMD: _mm256_maddubs_epi16 does 32 unsigned×signed
 * byte multiplies and 16 adds in ONE instruction.
 */
float vec_dot_q4_K_q8_K(const block_q4_K * GGML_RESTRICT x,
                         const block_q8_K * GGML_RESTRICT y,
                         int nb) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

#if defined(__AVX2__)
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Unpack scales and mins to 16-bit
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        // Min contribution: sum of (min * q8_bsums)
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(
            _mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        // Scale contribution
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    return hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#else
    // Scalar fallback
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        // Unpack scales (simplified)
        uint8_t sc[8], m[8];
        for (int j = 0; j < 4; j++) {
            sc[j]   = x[i].scales[j] & 63;
            m[j]    = x[i].scales[j+4] & 63;
        }
        for (int j = 4; j < 8; j++) {
            sc[j] = (x[i].scales[j+4] & 0x0f) | ((x[i].scales[j-4] >> 6) << 4);
            m[j]  = (x[i].scales[j+4] >> 4)    | ((x[i].scales[j-0] >> 6) << 4);
        }
        for (int j = 0; j < QK_K/2; j++) {
            int ql = x[i].qs[j] & 0x0f;
            int qh = x[i].qs[j] >> 4;
            int sb_l = j / 16;       // sub-block for low nibble
            int sb_h = j / 16 + 4;   // sub-block for high nibble (second 128 values)
            sumf += (d * sc[sb_l] * ql + dmin * m[sb_l]) * y[i].qs[j]
                  + (d * sc[sb_h] * qh + dmin * m[sb_h]) * y[i].qs[j + QK_K/2];
        }
    }
    return sumf;
#endif
}

/*
 * Q5_K × Q8_K dot product
 */
float vec_dot_q5_K_q8_K(const block_q5_K * GGML_RESTRICT x,
                         const block_q8_K * GGML_RESTRICT y,
                         int nb) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];

#if defined(__AVX2__)
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();
    float summs = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(
            _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        // Min contribution
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(
            _mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    return hsum_float_8(acc) + summs;

#else
    // Scalar fallback
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        uint8_t sc[8], m[8];
        for (int j = 0; j < 4; j++) {
            sc[j]   = x[i].scales[j] & 63;
            m[j]    = x[i].scales[j+4] & 63;
        }
        for (int j = 4; j < 8; j++) {
            sc[j] = (x[i].scales[j+4] & 0x0f) | ((x[i].scales[j-4] >> 6) << 4);
            m[j]  = (x[i].scales[j+4] >> 4)    | ((x[i].scales[j-0] >> 6) << 4);
        }
        for (int j = 0; j < QK_K/2; j++) {
            int l = x[i].qs[j] & 0x0f;
            int h = x[i].qs[j] >> 4;
            // Q5 adds high bit from qh
            int bit_idx_l = j;
            int bit_idx_h = j + QK_K/2;
            l |= (int)((x[i].qh[bit_idx_l/8] >> (bit_idx_l%8)) & 1) << 4;
            h |= (int)((x[i].qh[bit_idx_h/8] >> (bit_idx_h%8)) & 1) << 4;
            int sb_l = j / 16;
            int sb_h = j / 16 + 4;
            sumf += (d * sc[sb_l] * l + dmin * m[sb_l]) * y[i].qs[j]
                  + (d * sc[sb_h] * h + dmin * m[sb_h]) * y[i].qs[j + QK_K/2];
        }
    }
	return sumf;
#endif
}

/*
 * Q6_K × Q8_K dot product (the fast path from llama.cpp)
 * Q6_K uses 6-bit SIGNED quantization (range -32 to 31).
 * The key difference from Q4_K/Q5_K: values are unsigned 0-63 representing
 * signed -32..31, so we must subtract 32*sum(q8) from each partial product.
 * Uses _mm256_maddubs_epi16 with m32s (constant 32) for offset subtraction.
 * Scales are plain int8 (not packed 6-bit like Q4_K/Q5_K).
 */
float vec_dot_q6_K_q8_K(const block_q6_K * GGML_RESTRICT x,
	const block_q8_K * GGML_RESTRICT y,
	int nb) {

#if defined(__AVX2__)
	const __m256i m4 = _mm256_set1_epi8(0xF);
	const __m256i m2 = _mm256_set1_epi8(3);
	const __m256i m32s = _mm256_set1_epi8(32);

	__m256 acc = _mm256_setzero_ps();

	for (int i = 0; i < nb; ++i) {
		const float d = y[i].d * fp16_to_fp32(x[i].d);

		const uint8_t * GGML_RESTRICT q4 = x[i].ql;
		const uint8_t * GGML_RESTRICT qh = x[i].qh;
		const int8_t  * GGML_RESTRICT q8 = y[i].qs;

		const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

		__m256i sumi = _mm256_setzero_si256();

		int is = 0;

		for (int j = 0; j < QK_K/128; ++j) {
			const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
			const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
			const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
			const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
			is += 4;

			const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
			const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
			const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

			const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
			const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
			const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
			const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

			const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
			const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
			const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
			const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

			const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
			const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
			const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
			const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

			// Compute 32*sum(q8) for each pair of adjacent bytes
			__m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
			__m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
			__m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
			__m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

			// Unsigned q6 × signed q8 -> 16-bit products
			__m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
			__m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
			__m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
			__m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

			// Subtract offset: q6 values are 0..63 representing -32..31
			p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
			p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
			p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
			p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

			// Apply int8 scales (sign-extended to 16-bit)
			p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
			p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
			p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
			p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

			sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
			sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
		}

		acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
	}

	return hsum_float_8(acc);

#else
	// Scalar fallback
	float sumf = 0.0f;
	for (int i = 0; i < nb; ++i) {
		const float d = y[i].d * fp16_to_fp32(x[i].d);
		const uint8_t * GGML_RESTRICT ql = x[i].ql;
		const uint8_t * GGML_RESTRICT qh = x[i].qh;
		const int8_t  * GGML_RESTRICT q8 = y[i].qs;
		const int8_t  * sc = x[i].scales;

		int32_t sumi = 0;
		for (int j = 0; j < QK_K; j += 128) {
			for (int l = 0; l < 32; ++l) {
				// Reconstruct 6-bit value: lower 4 bits from ql, upper 2 bits from qh
				int q_low = (ql[j/2 + l] & 0xF) | (((qh[j/8 + l/4] >> (2*(l%4))) & 3) << 4);
				int q_high = (ql[j/2 + l + 32] & 0xF) | (((qh[j/8 + l/4] >> (2*(l%4) + 2)) & 3) << 4);
				// Subtract 32 to get signed value
				q_low -= 32;
				q_high -= 32;
				// Scale index: each scale covers 16 values
				int sc_idx = (j + l) / 16;
				sumi += sc[sc_idx] * (q_low * (int)q8[j + l] + q_high * (int)q8[j + l + 64]);
			}
			for (int l = 32; l < 64; ++l) {
				int q_low = ((ql[j/2 + l] >> 4) & 0xF) | (((qh[j/8 + l/4] >> (2*(l%4))) & 3) << 4);
				int q_high = ((ql[j/2 + l + 32] >> 4) & 0xF) | (((qh[j/8 + l/4] >> (2*(l%4) + 2)) & 3) << 4);
				q_low -= 32;
				q_high -= 32;
				int sc_idx = (j + l) / 16;
				sumi += sc[sc_idx] * (q_low * (int)q8[j + l] + q_high * (int)q8[j + l + 64]);
			}
		}
		sumf += d * (float)sumi;
	}
	return sumf;
#endif
}

/*
 * Q8_0 × Q8_0 dot product (simplest format, 32-element blocks)
 * x: Q8_0 weight blocks
 * y: Q8_0 quantized activation blocks
 * nb: number of 32-element blocks
 */
float vec_dot_q8_0_q8_0(const block_q8_0 * GGML_RESTRICT x,
                          const block_q8_0 * GGML_RESTRICT y,
                          int nb) {
    int ib = 0;
    float sumf = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
        const __m256i qx = _mm256_loadu_si256((const __m256i*)x[ib].qs);
        const __m256i qy = _mm256_loadu_si256((const __m256i*)y[ib].qs);
        const __m256 q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    sumf = hsum_float_8(acc);
#endif

    // Scalar tail
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[ib].qs[j] * (int)y[ib].qs[j];
        }
        sumf += sumi * (fp16_to_fp32(x[ib].d) * fp16_to_fp32(y[ib].d));
    }
    return sumf;
}

/* ---- Matrix-vector products ---- */

/*
 * Q4_K GEMV: out = weight * x
 * q8_buf: pre-allocated Q8_K buffer (inner_dim/256 blocks), reusable across calls
 */
void q4_k_gemv(const uint8_t * GGML_RESTRICT weight,
               const float * GGML_RESTRICT x,
               float * GGML_RESTRICT out,
               block_q8_K * GGML_RESTRICT q8_buf,
               int inner_dim, int outer_dim) {
    const int nb = inner_dim / QK_K;
    quantize_row_q8_K(x, q8_buf, inner_dim);
    const block_q4_K * w = (const block_q4_K *)weight;
    #pragma omp parallel for schedule(static) if(outer_dim > 64)
    for (int row = 0; row < outer_dim; row++) {
        out[row] = vec_dot_q4_K_q8_K(w + row * nb, q8_buf, nb);
    }
}

/*
 * Q5_K GEMV: out = weight * x
 * q8_buf: pre-allocated Q8_K buffer (inner_dim/256 blocks)
 */
void q5_k_gemv(const uint8_t * GGML_RESTRICT weight,
               const float * GGML_RESTRICT x,
               float * GGML_RESTRICT out,
               block_q8_K * GGML_RESTRICT q8_buf,
               int inner_dim, int outer_dim) {
    const int nb = inner_dim / QK_K;
    quantize_row_q8_K(x, q8_buf, inner_dim);
    const block_q5_K * w = (const block_q5_K *)weight;
    #pragma omp parallel for schedule(static) if(outer_dim > 64)
    for (int row = 0; row < outer_dim; row++) {
	out[row] = vec_dot_q5_K_q8_K(w + row * nb, q8_buf, nb);
	}
}

/*
 * Q6_K GEMV: out = weight * x
 * q8_buf: pre-allocated Q8_K buffer (inner_dim/256 blocks)
 */
void q6_k_gemv(const uint8_t * GGML_RESTRICT weight,
	const float * GGML_RESTRICT x,
	float * GGML_RESTRICT out,
	block_q8_K * GGML_RESTRICT q8_buf,
	int inner_dim, int outer_dim) {
	const int nb = inner_dim / QK_K;
	quantize_row_q8_K(x, q8_buf, inner_dim);
	const block_q6_K * w = (const block_q6_K *)weight;
	#pragma omp parallel for schedule(static) if(outer_dim > 64)
	for (int row = 0; row < outer_dim; row++) {
		out[row] = vec_dot_q6_K_q8_K(w + row * nb, q8_buf, nb);
	}
}

/*
 * Q8_0 GEMV: out = weight * x
 * Uses F32 activation directly (dequantize per 32-element block and multiply)
 * No Q8_K intermediate needed since Q8_0 is already int8 with per-block scale.
 */
void q8_0_gemv(const uint8_t * GGML_RESTRICT weight,
               const float * GGML_RESTRICT x,
               float * GGML_RESTRICT out,
               int inner_dim, int outer_dim) {
    const int nb = inner_dim / QK8_0;
    const block_q8_0 * w = (const block_q8_0 *)weight;

#if defined(__AVX2__)
    for (int row = 0; row < outer_dim; row++) {
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const block_q8_0 * blk = w + row * nb + b;
            float d = fp16_to_fp32(blk->d);
            const float * xv = x + b * QK8_0;

            // Load 32 int8 values and process in chunks of 8
            for (int j = 0; j < QK8_0; j += 8) {
                __m128i q8_raw = _mm_loadl_epi64((const __m128i*)(blk->qs + j));
                __m256i q8_32 = _mm256_cvtepi8_epi32(q8_raw);
                __m256 q8_f = _mm256_cvtepi32_ps(q8_32);
                __m256 xf = _mm256_loadu_ps(xv + j);
                __m256 prod = _mm256_mul_ps(q8_f, xf);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), prod, acc);
            }
        }
        out[row] = hsum_float_8(acc);
    }
#else
    for (int row = 0; row < outer_dim; row++) {
        float sumf = 0.0f;
        for (int b = 0; b < nb; b++) {
            const block_q8_0 * blk = w + row * nb + b;
            float d = fp16_to_fp32(blk->d);
            for (int j = 0; j < QK8_0; j++) {
                sumf += d * (float)blk->qs[j] * x[b * QK8_0 + j];
            }
        }
        out[row] = sumf;
    }
#endif
}
