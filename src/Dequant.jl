module Dequant

using ..QuantsData

export dequantize_iq2_xxs, dequantize_iq2_xs, dequantize_iq2_s, dequantize_iq3_xxs, dequantize_iq3_s,
       dequantize_iq4_xs, dequantize_iq4_nl, dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0

# --- Dequantization Logic ---

# --- IQ2_XXS Dequantization ---

function dequantize_iq2_xxs(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq2_xxs: 2 bytes scale (f16), 64 bytes quants
    # 256 elements per block
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    for i in 1:nb
        base = (i-1) * 66
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # 8 sub-blocks of 32 elements each
        for ib32 in 0:7
            qs_base = base + 2 + ib32 * 8
            
            # Read 8 bytes as 2 UInt32s
            aux32_1 = (UInt32(data[qs_base+1])) | (UInt32(data[qs_base+2]) << 8) | 
                      (UInt32(data[qs_base+3]) << 16) | (UInt32(data[qs_base+4]) << 24)
            aux32_2 = (UInt32(data[qs_base+5])) | (UInt32(data[qs_base+6]) << 8) | 
                      (UInt32(data[qs_base+7]) << 16) | (UInt32(data[qs_base+8]) << 24)
            
            db = d * (0.5f0 + (aux32_2 >> 28)) * 0.25f0
            
            # aux8 is the first 4 bytes (aux32_1)
            for l in 0:3
                grid_idx = ((aux32_1 >> (8*l)) & 0xFF) + 1
                grid_val = IQ2XXS_GRID[grid_idx]
                
                signs_idx = ((aux32_2 >> (7*l)) & 127) + 1
                signs = KSIGNS_IQ2XS[signs_idx]
                
                for j in 0:7
                    byte_val = ((grid_val >> (8 * j)) & 0xFF) % Int8 - 8
                    is_neg = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[(i-1)*256 + ib32*32 + l*8 + j + 1] = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ2_XS Dequantization ---

function dequantize_iq2_xs(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq2_xs: 2 bytes scale (f16), 64 bytes quants (uint16_t[32]), 8 bytes scales
    # Total = 74 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 74
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # 8 sub-blocks of 32
        for ib32 in 0:7
            db0 = d * (0.5f0 + (data[base + 66 + ib32 + 1] & 0x0F)) * 0.25f0
            db1 = d * (0.5f0 + (data[base + 66 + ib32 + 1] >> 4)) * 0.25f0
            
            for l in 0:3
                qs_idx = base + 2 + (ib32*4 + l)*2
                v = (UInt16(data[qs_idx+2]) << 8) | UInt16(data[qs_idx+1])
                
                grid_idx = (v & 511) + 1
                grid_val = IQ2XS_GRID[grid_idx]
                
                signs_idx = (v >> 9) + 1
                signs = KSIGNS_IQ2XS[signs_idx]
                
                db = (l < 2) ? db0 : db1
                
                for j in 0:7
                    byte_val = ((grid_val >> (8 * j)) & 0xFF) % Int8 - 8
                    is_neg = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[(i-1)*256 + ib32*32 + l*8 + j + 1] = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ2_S Dequantization ---

function dequantize_iq2_s(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq2_s: 2 bytes scale (f16), 64 bytes qs, 8 bytes qh, 8 bytes scales
    # Total = 82 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 82
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # signs start at data[base + 64 + 2 + 1] ??? no, wait.
        # qs[64], qh[8], scales[8]
        qs = @view data[base + 3:base + 2 + 64]
        qh = @view data[base + 3 + 64:base + 2 + 64 + 8]
        scales = @view data[base + 3 + 64 + 8:base + 2 + 64 + 8 + 8]
        # wait, signs in iq2_s? ggml-quants.c says: const uint8_t * signs = qs + QK_K/8;
        # Wait, if qs is 64 bytes (QK_K/4), and signs is 32 bytes (QK_K/8)...
        # block_iq2_s struct: uint8_t qs[QK_K/4], uint8_t qh[QK_K/32], uint8_t scales[QK_K/32].
        # In dequantize_row_iq2_s: signs = qs + QK_K/8;
        # That means signs are the second half of qs?
        # QK_K/4 = 64. QK_K/8 = 32. So signs = qs[33:64].
        
        for ib32 in 0:7
            db0 = d * (0.5f0 + (scales[ib32+1] & 0x0F)) * 0.25f0
            db1 = d * (0.5f0 + (scales[ib32+1] >> 4)) * 0.25f0
            
            for l in 0:3
                # grid index: qs[l] | (qh[ib32] << (8-2*l) & 0x300)
                # wait, which qs? qs for this ib32?
                # ib32 loop in C: qs is fixed? No, qs is x[i].qs.
                # y[j] loop: y += 8. l loop: l < 4.
                # Grid idx calculation: grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                # signs calculation: signs[l] & kmask_iq2xs[j]
                # Wait, signs is qs + 256/8 = qs + 32. 
                
                # So if we are in ib32, which part of qs?
                # ib32 goes 0..7. Each ib32 consumes 4 'y' increments of 8 elements = 32 elements.
                # So in total 32 loops of l*8? No. 8 ib32 * 4 l = 32.
                # So qs index should be ib32*4 + l.
                
                q_idx = ib32*4 + l + 1
                grid_idx = Int(qs[q_idx]) | (Int(qh[ib32+1]) << (8 - 2*l) & 0x300) + 1
                grid_val = IQ2S_GRID[grid_idx]
                
                sign_val = qs[q_idx + 32] # signs = qs + 32
                db = (l < 2) ? db0 : db1
                
                for j in 0:7
                    byte_val = ((grid_val >> (8 * j)) & 0xFF) % Int8 - 8
                    is_neg = (sign_val & KMASK_IQ2XS[j+1]) != 0
                    y[(i-1)*256 + ib32*32 + l*8 + j + 1] = db * Float32(byte_val) * (is_neg ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ3_XXS Dequantization ---

function dequantize_iq3_xxs(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq3_xxs: 2 bytes scale (f16), 96 bytes quants (3*QK_K/8)
    # Total = 98 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 98
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # qs: 64 bytes? No, qs is x[i].qs. 
        # scales_and_signs = qs + 256/4 = qs + 64.
        # QK_K/4 = 64. 64 + 32 (scales) = 96. Total 98. Correct.
        
        for ib32 in 0:7
            # scales_and_signs is at data[base + 2 + 64]
            # aux32 = memcpy from scales_and_signs + 4*ib32. 
            # wait, ib32 goes 0..7. 4*7 = 28. fits in 32 bytes scales.
            ss_base = base + 2 + 64 + 4*ib32
            aux32 = (UInt32(data[ss_base+1])) | (UInt32(data[ss_base+2]) << 8) | 
                    (UInt32(data[ss_base+3]) << 16) | (UInt32(data[ss_base+4]) << 24)
            
            db = d * (0.5f0 + (aux32 >> 28)) * 0.5f0
            
            for l in 0:3
                signs_idx = ((aux32 >> (7*l)) & 127) + 1
                signs = KSIGNS_IQ2XS[signs_idx]
                
                # qs indices for this ib32 and l
                # ib32*8 + 2*l
                # Wait, qs is at base + 2.
                # ib32 goes 0..7. l goes 0..3. Total 32 l-increments.
                # Each l increment consumes 2 qs values. Total 64 qs values. fits.
                q_base = base + 2 + ib32*8 + 2*l
                grid1 = IQ3XXS_GRID[data[q_base+1] + 1]
                grid2 = IQ3XXS_GRID[data[q_base+2] + 1]
                
                for j in 0:3
                    y_idx = (i-1)*256 + ib32*32 + l*8 + j + 1
                    byte_val1 = ((grid1 >> (8 * j)) & 0xFF) % Int8 - 4
                    is_neg1 = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[y_idx] = db * Float32(byte_val1) * (is_neg1 ? -1.0f0 : 1.0f0)
                    
                    byte_val2 = ((grid2 >> (8 * j)) & 0xFF) % Int8 - 4
                    is_neg2 = (signs & KMASK_IQ2XS[j+5]) != 0
                    y[y_idx + 4] = db * Float32(byte_val2) * (is_neg2 ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

# --- IQ3_S Dequantization ---

function dequantize_iq3_s(data::AbstractVector{UInt8}, num_elements::Int)
    # block_iq3_s: 2 bytes d, 64 bytes qs, 8 bytes qh, 32 bytes signs, 4 bytes scales
    # Total = 110 bytes per 256 elements
    @assert num_elements % 256 == 0
    nb = num_elements ÷ 256
    y = Vector{Float32}(undef, num_elements)
    
    for i in 1:nb
        base = (i-1) * 110
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        
        # ib32 += 2 loop means we process 64 elements at a time
        for ib32 in 0:2:7
            scales_val = data[base + 2 + 64 + 8 + 32 + (ib32 ÷ 2) + 1]
            db1 = d * (1.0f0 + 2.0f0 * (scales_val & 0x0F))
            db2 = d * (1.0f0 + 2.0f0 * (scales_val >> 4))
            
            # First 32 elements (db1)
            for l in 0:3
                q_idx = ib32 * 8 + 2 * l + 1 # wait, ib32*8? 
                # C code: qs += 8 AFTER the first l loop.
                # So the first l loop uses qs indices 0..7.
                # The second l loop uses qs indices 8..15.
                # But ib32 advances by 2. So ib32=0 uses 0..15, ib32=2 uses 16..31?
                # QK_K/4 = 64 bytes for qs. 256 elements.
                # Total 32 l-increments for the whole block?
                # ib32 = 0, 2, 4, 6. For each ib32, two l loops of 4. Total 4 * 2 * 4 = 32 l loops. 32 * 8 = 256. Correct.
                
                # Global qs index for ib32 and l
                qs_off = ib32 * 8
                qh_val = data[base + 2 + 64 + (ib32) + 1]
                signs_off = base + 2 + 64 + 8 + ib32 * 4
                
                # First l loop (db1)
                q_base = base + 2 + qs_off + 2*l
                grid1 = IQ3S_GRID[(Int(data[q_base+1]) | (Int(qh_val << (8 - 2*l)) & 256)) + 1]
                grid2 = IQ3S_GRID[(Int(data[q_base+2]) | (Int(qh_val << (7 - 2*l)) & 256)) + 1]
                
                signs = data[signs_off + l + 1]
                
                for j in 0:3
                    y_idx = (i-1)*256 + ib32*32 + l*8 + j + 1
                    byte_val1 = ((grid1 >> (8 * j)) & 0xFF) % Int8 - 1
                    is_neg1 = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[y_idx] = db1 * Float32(byte_val1) * (is_neg1 ? -1.0f0 : 1.0f0)
                    
                    byte_val2 = ((grid2 >> (8 * j)) & 0xFF) % Int8 - 1
                    is_neg2 = (signs & KMASK_IQ2XS[j+5]) != 0
                    y[y_idx + 4] = db1 * Float32(byte_val2) * (is_neg2 ? -1.0f0 : 1.0f0)
                end
            end
            
            # Second 32 elements (db2) - ib32+1? 
            # In C: ib32 increment is 2. it uses scales[ib32/2] for both db1 and db2.
            # db1 = d * (1 + 2*(scales & 0xf)), db2 = d * (1 + 2*(scales >> 4))
            # The second l loop uses qs += 8.
            
            for l in 0:3
                qs_off = (ib32 + 1) * 8
                qh_val = data[base + 2 + 64 + (ib32 + 1) + 1]
                signs_off = base + 2 + 64 + 8 + (ib32 + 1) * 4
                
                q_base = base + 2 + qs_off + 2*l
                grid1 = IQ3S_GRID[(Int(data[q_base+1]) | (Int(qh_val << (8 - 2*l)) & 256)) + 1]
                grid2 = IQ3S_GRID[(Int(data[q_base+2]) | (Int(qh_val << (7 - 2*l)) & 256)) + 1]
                
                signs = data[signs_off + l + 1]
                
                for j in 0:3
                    y_idx = (i-1)*256 + (ib32+1)*32 + l*8 + j + 1
                    byte_val1 = ((grid1 >> (8 * j)) & 0xFF) % Int8 - 1
                    is_neg1 = (signs & KMASK_IQ2XS[j+1]) != 0
                    y[y_idx] = db2 * Float32(byte_val1) * (is_neg1 ? -1.0f0 : 1.0f0)
                    
                    byte_val2 = ((grid2 >> (8 * j)) & 0xFF) % Int8 - 1
                    is_neg2 = (signs & KMASK_IQ2XS[j+5]) != 0
                    y[y_idx + 4] = db2 * Float32(byte_val2) * (is_neg2 ? -1.0f0 : 1.0f0)
                end
            end
        end
    end
    return y
end

function dequantize_q8_0(data::AbstractVector{UInt8}, num_elements::Int)
    # block_q8_0: 2 bytes scale (f16), 32 bytes quants (int8)
    # 32 elements per block
    @assert num_elements % 32 == 0
    nb = num_elements ÷ 32
    y = Vector{Float32}(undef, num_elements)
    for i in 1:nb
        base = (i-1) * 34
        d_raw = (UInt16(data[base+2]) << 8) | UInt16(data[base+1])
        d = Float32(reinterpret(Float16, d_raw))
        for j in 1:32
            q_u8 = data[base+2+j]
            q = Float32(reinterpret(Int8, q_u8))
            y[(i-1)*32 + j] = d * q
        end
    end
    return y
end

# --- Q4_K Dequantization ---

function dequantize_q4_k(data::AbstractVector{UInt8}, num_elements::Int)
    nb = num_elements ÷ 256
    # block_q4_K:
    #   d, dmin: 2*2 = 4 bytes (ggml_half)
    #   scales: 12 bytes
    #   qs: 128 bytes
    # Total = 4 + 12 + 128 = 144 bytes per block.
    block_size = 144
    weights = Vector{Float32}(undef, num_elements)
    
    for i in 0:(nb - 1)
        offset = i * block_size + 1
        d = Float32(reinterpret(Float16, data[offset:(offset + 1)])[1])
        dmin = Float32(reinterpret(Float16, data[(offset + 2):(offset + 3)])[1])
        
        scales = @view data[(offset + 4):(offset + 15)]
        qs = @view data[(offset + 16):(offset + 143)]
        
        idx_base = i * 256
        
        for j in 0:3 # 4 super-groups of 64
            is_idx = 2 * j
            
            # sc1, m1 (first 32 elements)
            sc1, m1 = if is_idx < 4
                UInt8(scales[is_idx + 1] & 63), UInt8(scales[is_idx + 5] & 63)
            else
                UInt8((scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4)),
                UInt8((scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4))
            end
            
            # sc2, m2 (next 32 elements)
            sc2, m2 = if (is_idx + 1) < 4
                UInt8(scales[is_idx + 2] & 63), UInt8(scales[is_idx + 6] & 63)
            else
                UInt8((scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4)),
                UInt8((scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4))
            end
            
            d1, min1 = d * sc1, dmin * m1
            d2, min2 = d * sc2, dmin * m2
            
            for l in 0:31
                ql_val = qs[j * 32 + l + 1]
                weights[idx_base + j * 64 + l + 1] = d1 * Float32(ql_val & 0x0f) - min1
                weights[idx_base + j * 64 + 32 + l + 1] = d2 * Float32(ql_val >> 4) - min2
            end
        end
    end
    return weights
end

# --- Q5_K Dequantization ---

function dequantize_q5_k(data::AbstractVector{UInt8}, num_elements::Int)
    nb = num_elements ÷ 256
    # block_q5_K:
    #   d: 2 bytes
    #   dmin: 2 bytes
    #   scales: 12 bytes
    #   qh: 32 bytes
    #   qs: 128 bytes
    # Total = 176 bytes per block.
    block_size = 176
    weights = Vector{Float32}(undef, num_elements)

    for i in 0:(nb - 1)
        offset = i * block_size + 1
        
        # d and dmin are ggml_half (Float16)
        d = Float32(reinterpret(Float16, data[offset:(offset + 1)])[1])
        dmin = Float32(reinterpret(Float16, data[(offset + 2):(offset + 3)])[1])
        
        scales = @view data[(offset + 4):(offset + 15)]
        qh = @view data[(offset + 16):(offset + 47)]
        qs = @view data[(offset + 48):offset + 175]

        u1 = UInt8(1)
        u2 = UInt8(2)
        
        idx_base = i * 256
        
        for j in 0:3 # 4 blocks of 64
            is_idx = 2 * j
            
            # sc1, m1 (first 32 elements)
            sc1, m1 = if is_idx < 4
                scales[is_idx + 1] & 63, scales[is_idx + 5] & 63
            else
                (scales[is_idx + 5] & 0x0f) | ((scales[is_idx - 3] >> 6) << 4),
                (scales[is_idx + 5] >> 4) | ((scales[is_idx + 1] >> 6) << 4)
            end
            
            # sc2, m2 (next 32 elements)
            sc2, m2 = if (is_idx + 1) < 4
                scales[is_idx + 2] & 63, scales[is_idx + 6] & 63
            else
                (scales[is_idx + 6] & 0x0f) | ((scales[is_idx - 2] >> 6) << 4),
                (scales[is_idx + 6] >> 4) | ((scales[is_idx + 2] >> 6) << 4)
            end
            
            d1, min1 = d * sc1, dmin * m1
            d2, min2 = d * sc2, dmin * m2
            
            u1 = UInt8(1 << (2 * j))
            u2 = UInt8(1 << (2 * j + 1))

            for l in 0:31
                ql_val = qs[j * 32 + l + 1]
                qh_val = qh[l + 1]
                weights[idx_base + j * 64 + l + 1]  = d1 * Float32((ql_val & 0x0f) + (qh_val & u1 != 0 ? 16 : 0)) - min1
                weights[idx_base + j * 64 + l + 33] = d2 * Float32((ql_val >> 4)  + (qh_val & u2 != 0 ? 16 : 0)) - min2
            end
        end
    end
    return weights
end

# --- Q2_K Dequantization ---

function dequantize_q2_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q2_K:
    #   scales[16], qs[64], d, dmin
    # Total = 16 + 64 + 2 + 2 = 84 bytes per block.
    
    for i in 1:nb
        base = (i-1) * 84
        scales = @view data[base+1:base+16]
        qs = @view data[base+17:base+80]
        d = Float32(reinterpret(Float16, (UInt16(data[base+82]) << 8) | UInt16(data[base+81])))
        m = Float32(reinterpret(Float16, (UInt16(data[base+84]) << 8) | UInt16(data[base+83])))
        
        is_idx = 1
        q_off = 0
        for n in 0:128:255
            shift = 0
            for j in 0:3
                sc1 = scales[is_idx]
                dl1 = d * (sc1 & 0x0F); ml1 = m * (sc1 >> 4)
                for l in 0:15
                    y[(i-1)*256 + n + j*16 + l + 1] = dl1 * ((qs[q_off + l + 1] >> shift) & 3) - ml1
                end
                
                sc2 = scales[is_idx+1]
                dl2 = d * (sc2 & 0x0F); ml2 = m * (sc2 >> 4)
                for l in 0:15
                    y[(i-1)*256 + n + j*16 + l + 17] = dl2 * ((qs[q_off + l + 17] >> shift) & 3) - ml2
                end
                
                shift += 2
                is_idx += 2
            end
            q_off += 32
        end
    end
    return y
end

# --- Q3_K Dequantization ---

function dequantize_q3_k(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 256 == 0
    k = num_elements
    nb = k ÷ 256
    y = Vector{Float32}(undef, k)
    
    # block_q3_K:
    #   hmask[32], qs[64], scales[12], d
    # Total = 32 + 64 + 12 + 2 = 110 bytes per block.
    
    for i in 1:nb
        base = (i-1) * 110
        hmask = @view data[base+1:base+32]
        qs = @view data[base+33:base+96]
        scales_raw = @view data[base+97:base+108]
        d_all = Float32(reinterpret(Float16, (UInt16(data[base+110]) << 8) | UInt16(data[base+109])))
        
        # Repack scales (12 bytes -> 16 scales)
        a0 = UInt32(scales_raw[1]) | (UInt32(scales_raw[2]) << 8) | (UInt32(scales_raw[3]) << 16) | (UInt32(scales_raw[4]) << 24)
        a1 = UInt32(scales_raw[5]) | (UInt32(scales_raw[6]) << 8) | (UInt32(scales_raw[7]) << 16) | (UInt32(scales_raw[8]) << 24)
        a2 = UInt32(scales_raw[9]) | (UInt32(scales_raw[10]) << 8) | (UInt32(scales_raw[11]) << 16) | (UInt32(scales_raw[12]) << 24)
        
        kmask1 = 0x03030303
        kmask2 = 0x0f0f0f0f
        
        s0 = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4)
        s1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4)
        s2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4)
        s3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4)
        
        repacked_scales = Vector{Int8}(undef, 16)
        for k_sc in 0:3
            repacked_scales[k_sc*4 + 1] = (s0 >> (8*k_sc) & 0xFF) % Int8
            repacked_scales[k_sc*4 + 2] = (s1 >> (8*k_sc) & 0xFF) % Int8
            repacked_scales[k_sc*4 + 3] = (s2 >> (8*k_sc) & 0xFF) % Int8
            repacked_scales[k_sc*4 + 4] = (s3 >> (8*k_sc) & 0xFF) % Int8
        end
        
        is_idx = 1
        q_off = 0
        m_bit = UInt8(1)
        for n in 0:128:255
            shift = 0
            for j in 0:3
                dl1 = d_all * (Float32(repacked_scales[is_idx]) - 32.0f0)
                for l in 0:15
                    y[(i-1)*256 + n + l + 1] = dl1 * (Float32((qs[q_off + l + 1] >> shift) & 3) - ((hmask[l + 1] & m_bit) != 0 ? 0.0f0 : 4.0f0))
                end
                is_idx += 1
                
                dl2 = d_all * (Float32(repacked_scales[is_idx]) - 32.0f0)
                for l in 0:15
                    y[(i-1)*256 + n + l + 17] = dl2 * (Float32((qs[q_off + l + 17] >> shift) & 3) - ((hmask[l + 17] & m_bit) != 0 ? 0.0f0 : 4.0f0))
                end
                is_idx += 1
                
                shift += 2
                m_bit <<= 1
            end
            q_off += 32
        end
    end
    return y
end


# --- Q6_K Dequantization ---

function dequantize_q6_k(data::AbstractVector{UInt8}, num_elements::Int)
    nb = num_elements ÷ 256
    # block_q6_K:
    #   ql: 128 bytes (QK_K/2)
    #   qh: 64 bytes (QK_K/4)
    #   scales: 16 bytes (QK_K/16)
    #   d: 2 bytes (ggml_half)
    # Total = 128 + 64 + 16 + 2 = 210 bytes.
    block_size = 210
    weights = Vector{Float32}(undef, num_elements)

    for i in 0:(nb - 1)
        offset = i * block_size + 1
        ql = @view data[offset:(offset + 127)]
        qh = @view data[(offset + 128):(offset + 191)]
        sc = reinterpret(Int8, data[(offset + 192):(offset + 207)])
        d = Float32(reinterpret(Float16, data[(offset + 208):(offset + 209)])[1])

        idx_base = i * 256
        
        for n_sg in 0:1
            ql_sg = @view ql[(n_sg * 64 + 1):(n_sg * 64 + 64)]
            qh_sg = @view qh[(n_sg * 32 + 1):(n_sg * 32 + 32)]
            sc_sg = @view sc[(n_sg * 8 + 1):(n_sg * 8 + 8)]
            
            for l in 0:31
                is_idx = l ÷ 16
                
                qh_val = qh_sg[l + 1]
                
                q1 = Int8((ql_sg[l + 1] & 0x0f) | ((qh_val & 0x03) << 4)) - Int8(32)
                q2 = Int8((ql_sg[l + 32 + 1] & 0x0f) | (((qh_val >> 2) & 0x03) << 4)) - Int8(32)
                q3 = Int8((ql_sg[l + 1] >> 4) | (((qh_val >> 4) & 0x03) << 4)) - Int8(32)
                q4 = Int8((ql_sg[l + 32 + 1] >> 4) | (((qh_val >> 6) & 0x03) << 4)) - Int8(32)
                
                weights[idx_base + n_sg * 128 + l + 1]       = d * Float32(sc_sg[is_idx + 1]) * Float32(q1)
                weights[idx_base + n_sg * 128 + l + 32 + 1]  = d * Float32(sc_sg[is_idx + 2 + 1]) * Float32(q2)
                weights[idx_base + n_sg * 128 + l + 64 + 1]  = d * Float32(sc_sg[is_idx + 4 + 1]) * Float32(q3)
                weights[idx_base + n_sg * 128 + l + 96 + 1]  = d * Float32(sc_sg[is_idx + 6 + 1]) * Float32(q4)
            end
        end
    end
    return weights
end

function dequantize_iq4_xs(data::AbstractVector{UInt8}, num_elements::Int)
    nb = num_elements ÷ 256
    # block_iq4_xs:
    #   d: 2 bytes (ggml_half)
    #   scales_h: 2 bytes (uint16_t)
    #   scales_l: 4 bytes (uint8_t[4])
    #   qs: 128 bytes (uint8_t[128])
    # Total = 2 + 2 + 4 + 128 = 136 bytes per block.
    block_size = 136
    weights = Vector{Float32}(undef, num_elements)
    
    for i in 0:(nb - 1)
        offset = i * block_size + 1
        d = Float32(reinterpret(Float16, data[offset:(offset + 1)])[1])
        scales_h = UInt16(data[offset + 2]) | (UInt16(data[offset + 3]) << 8)
        
        idx_base = i * 256
        
        for ib in 0:7
            byte_idx = offset + 4 + ib ÷ 2
            shift_amt = 4 * (ib % 2)
            byte_val = data[byte_idx]
            low_bits = (byte_val >> shift_amt) & 0x0f
            shift_h = 2 * ib
            shifted = scales_h >> shift_h
            high_bits = shifted & 0x03
            ls = low_bits | (high_bits << 4)
            dl = d * Float32(Int32(ls) - 32)
            
            for j in 0:15
                q_val = data[offset + 8 + ib * 16 + j]
                weights[idx_base + ib * 32 + j + 1] = dl * Float32(KVALUES_IQ4NL[(q_val & 0x0f) + 1])
                weights[idx_base + ib * 32 + j + 16 + 1] = dl * Float32(KVALUES_IQ4NL[(q_val >> 4) + 1])
            end
        end
    end
    return weights
end

"""
    dequantize_iq4_nl(data, num_elements) -> Vector{Float32}

Dequantize IQ4_NL quantized weights.

Block layout (block_iq4_nl, QK4_NL == 32):
  d     : ggml_bfloat16 / f16  (2 bytes)
  qs[16]: uint8_t              (16 bytes, packed 4-bit nibbles)
  Total = 18 bytes per block of 32 elements.

Each block uses a single global scale d; each weight = d * KVALUES_IQ4NL[nibble].
Matches llama.cpp `dequantize_row_iq4_nl`.

Reference: llama.cpp ggml-quants.c + gguf-py gguf/quants.py (IQ4_NL.kvalues).
"""
function dequantize_iq4_nl(data::AbstractVector{UInt8}, num_elements::Int)
    @assert num_elements % 32 == 0
    nb = num_elements ÷ 32
    block_size = 18  # 2 (d) + 16 (qs)
    weights = Vector{Float32}(undef, num_elements)

    for i in 0:(nb - 1)
        offset = i * block_size + 1
        d = Float32(reinterpret(Float16, data[offset:(offset + 1)])[1])
        qs = @view data[offset + 2:(offset + 17)]
        idx_base = i * 32 + 1

        for j in 0:15
            weights[idx_base + j]     = d * Float32(KVALUES_IQ4NL[(qs[j + 1] & 0x0f) + 1])
            weights[idx_base + j + 16] = d * Float32(KVALUES_IQ4NL[(qs[j + 1] >> 4) + 1])
        end
    end
    return weights
end

"""
    dequantize_iq4_nl_into!(out, data, block_offset)

In-place dequantization of one IQ4_NL block into `out` (32 elements),
matching llama.cpp `dequantize_row_iq4_nl` element order.
"""
function dequantize_iq4_nl_into!(out::Vector{Float32}, data::AbstractVector{UInt8}, block_offset::Int)
    @assert length(out) == 32
    d = Float32(reinterpret(Float16, data[block_offset:block_offset + 1])[1])
    qs = @view data[block_offset + 2:block_offset + 17]
    @inbounds for j in 0:15
        out[j + 1]     = d * Float32(KVALUES_IQ4NL[(qs[j + 1] & 0x0f) + 1])
        out[j + 17]    = d * Float32(KVALUES_IQ4NL[(qs[j + 1] >> 4) + 1])
    end
    return out
end

end # module
