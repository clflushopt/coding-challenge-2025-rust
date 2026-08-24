use crate::parameter::Parameter;
use std::collections::HashMap;

const INDEX_EMPTY: u8 = 0;
const INDEX_EXACT_COMPRESSED: u8 = 1;
const INDEX_HYBRID: u8 = 2;
const INDEX_DENSE_BITMAP: u8 = 3;

pub fn build_idx(parameter: &Parameter, data: &[i32]) -> Box<[u8]> {
    if data.is_empty() {
        return vec![INDEX_EMPTY].into_boxed_slice();
    }

    let mut freq_map: HashMap<i32, u64> = HashMap::new();
    for &val in data {
        *freq_map.entry(val).or_insert(0) += 1;
    }

    let distinct_count = freq_map.len();
    let ratio = parameter.factor_skip as f64 / parameter.factor_size as f64;
    let total_elements = data.len();

    let min_val = *data.iter().min().unwrap();
    let max_val = *data.iter().max().unwrap();

    let range_u64 = (max_val as i64 - min_val as i64 + 1) as u64;
    if range_u64 <= 262144 && distinct_count as f64 / range_u64 as f64 > 0.1 {
        let bitmap_size = (range_u64 as usize + 7) / 8;
        if bitmap_size < 32768 && ratio > 3.0 {
            return build_dense_bitmap(&freq_map, min_val, max_val, range_u64 as usize);
        }
    }

    if distinct_count <= 200 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    if distinct_count <= 600 && ratio > 10.0 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    if distinct_count <= 1200 && ratio > 25.0 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    let mut sorted_freq: Vec<_> = freq_map.iter().map(|(k, v)| (*k, *v)).collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(&a.1));

    let top_10_pct_count = (distinct_count / 10).max(10).min(100);
    let top_concentration: u64 = sorted_freq
        .iter()
        .take(top_10_pct_count)
        .map(|(_, c)| c)
        .sum();
    let concentration_ratio = top_concentration as f64 / total_elements as f64;

    let top_k = calculate_adaptive_top_k(ratio, distinct_count, concentration_ratio);
    let bloom_size =
        calculate_adaptive_bloom_size(ratio, distinct_count, data.len(), concentration_ratio);

    build_hybrid_index(&freq_map, data, top_k, bloom_size, min_val, max_val)
}

fn calculate_adaptive_top_k(ratio: f64, distinct_count: usize, concentration: f64) -> usize {
    let base_k = if ratio > 25.0 {
        200
    } else if ratio > 15.0 {
        150
    } else if ratio > 8.0 {
        100
    } else if ratio > 4.0 {
        60
    } else if ratio > 2.0 {
        30
    } else {
        15
    };

    let concentration_multiplier = if concentration > 0.5 {
        1.5
    } else if concentration > 0.3 {
        1.3
    } else if concentration > 0.15 {
        1.15
    } else {
        1.0
    };

    let adjusted_k = (base_k as f64 * concentration_multiplier) as usize;
    adjusted_k.min(distinct_count / 2).max(10)
}

fn calculate_adaptive_bloom_size(
    ratio: f64,
    distinct_count: usize,
    block_size: usize,
    concentration: f64,
) -> usize {
    let base_fpr = if concentration > 0.4 {
        0.015
    } else if ratio > 20.0 {
        0.003
    } else if ratio > 10.0 {
        0.007
    } else if ratio > 5.0 {
        0.015
    } else if ratio > 2.5 {
        0.04
    } else {
        0.08
    };

    let ln2_squared = 0.4804530139182014;
    let optimal_bits =
        (-1.0 * distinct_count as f64 * (base_fpr as f64).ln() / ln2_squared).ceil() as usize;
    let optimal_bytes = (optimal_bits + 7) / 8;

    let min_size = if ratio > 15.0 { 768 } else { 512 };
    let max_size = if ratio > 20.0 {
        6144
    } else if ratio > 10.0 {
        4096
    } else {
        3072
    };

    optimal_bytes.clamp(min_size, max_size)
}

fn encode_varint(value: u64, buffer: &mut Vec<u8>) {
    let mut v = value;
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buffer.push(byte);
        if v == 0 {
            break;
        }
    }
}

fn decode_varint(data: &[u8], offset: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0;

    loop {
        if *offset >= data.len() {
            return None;
        }

        let byte = data[*offset];
        *offset += 1;

        result |= ((byte & 0x7F) as u64) << shift;

        if byte & 0x80 == 0 {
            break;
        }

        shift += 7;
        if shift >= 64 {
            return None;
        }
    }

    Some(result)
}

fn build_dense_bitmap(
    freq_map: &HashMap<i32, u64>,
    min_val: i32,
    max_val: i32,
    range: usize,
) -> Box<[u8]> {
    let mut buffer = Vec::new();
    buffer.push(INDEX_DENSE_BITMAP);

    buffer.extend_from_slice(&min_val.to_le_bytes());
    buffer.extend_from_slice(&max_val.to_le_bytes());

    let bitmap_bytes = (range + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_bytes];

    for &key in freq_map.keys() {
        let offset = (key as i64 - min_val as i64) as usize;
        bitmap[offset / 8] |= 1 << (offset % 8);
    }

    buffer.extend_from_slice(&bitmap);

    let mut sorted_keys: Vec<i32> = freq_map.keys().copied().collect();
    sorted_keys.sort_unstable();

    for key in sorted_keys {
        let count = freq_map[&key];
        encode_varint(count, &mut buffer);
    }

    buffer.into_boxed_slice()
}

fn build_exact_compressed(freq_map: &HashMap<i32, u64>, min_val: i32, max_val: i32) -> Box<[u8]> {
    let mut entries: Vec<_> = freq_map.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|&(k, _)| k);

    let mut buffer = Vec::new();
    buffer.push(INDEX_EXACT_COMPRESSED);

    buffer.extend_from_slice(&min_val.to_le_bytes());
    buffer.extend_from_slice(&max_val.to_le_bytes());
    buffer.extend_from_slice(&(entries.len() as u32).to_le_bytes());

    let mut prev_key = 0i32;
    for (i, (key, count)) in entries.iter().enumerate() {
        if i == 0 {
            buffer.extend_from_slice(&key.to_le_bytes());
        } else {
            let delta = key.wrapping_sub(prev_key);
            buffer.extend_from_slice(&delta.to_le_bytes());
        }
        prev_key = *key;

        encode_varint(*count, &mut buffer);
    }

    buffer.into_boxed_slice()
}

fn build_hybrid_index(
    freq_map: &HashMap<i32, u64>,
    data: &[i32],
    top_k: usize,
    bloom_size: usize,
    min_val: i32,
    max_val: i32,
) -> Box<[u8]> {
    let mut sorted_freq: Vec<_> = freq_map.iter().collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(a.1));

    let mut top_entries: Vec<_> = sorted_freq
        .iter()
        .take(top_k.min(freq_map.len()))
        .map(|(&k, &v)| (k, v))
        .collect();

    top_entries.sort_unstable_by_key(|&(k, _)| k);

    let top_keys: std::collections::HashSet<i32> = top_entries.iter().map(|(k, _)| *k).collect();

    let remaining_distinct = freq_map.len().saturating_sub(top_entries.len());
    let m = bloom_size * 8;
    let n = remaining_distinct.max(1);
    let optimal_k = ((m as f64 / n as f64) * std::f64::consts::LN_2).round() as u8;
    let num_hashes = optimal_k.clamp(6, 11);

    let mut bloom = vec![0u8; bloom_size];

    for &val in data {
        if top_keys.contains(&val) {
            continue;
        }

        for i in 0..num_hashes {
            let hash = hash_value(val, i as u32);
            let bit_pos = (hash as usize) % (bloom_size * 8);
            bloom[bit_pos / 8] |= 1 << (bit_pos % 8);
        }
    }

    let mut buffer = Vec::new();
    buffer.push(INDEX_HYBRID);

    buffer.extend_from_slice(&min_val.to_le_bytes());
    buffer.extend_from_slice(&max_val.to_le_bytes());

    buffer.push(num_hashes);
    buffer.extend_from_slice(&(top_entries.len() as u32).to_le_bytes());

    let mut prev_key = 0i32;
    for (i, (key, count)) in top_entries.iter().enumerate() {
        if i == 0 {
            buffer.extend_from_slice(&key.to_le_bytes());
        } else {
            let delta = key.wrapping_sub(prev_key);
            buffer.extend_from_slice(&delta.to_le_bytes());
        }
        prev_key = *key;

        encode_varint(*count, &mut buffer);
    }

    buffer.extend_from_slice(&bloom);

    buffer.into_boxed_slice()
}

pub fn query_idx(parameter: &Parameter, index: &[u8], query: &i32) -> Option<u64> {
    if index.is_empty() {
        return None;
    }

    let index_type = index[0];

    match index_type {
        INDEX_EMPTY => None,

        INDEX_DENSE_BITMAP => {
            if index.len() < 9 {
                return None;
            }

            let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
            let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

            if *query < min_val || *query > max_val {
                return Some(0);
            }

            let range = (max_val as i64 - min_val as i64 + 1) as usize;
            let bitmap_bytes = (range + 7) / 8;

            if index.len() < 9 + bitmap_bytes {
                return None;
            }

            let offset_in_range = (*query as i64 - min_val as i64) as usize;
            let byte_pos = offset_in_range / 8;
            let bit_pos = offset_in_range % 8;

            if (index[9 + byte_pos] & (1 << bit_pos)) == 0 {
                return Some(0);
            }

            let mut count_before = 0;
            for i in 0..offset_in_range {
                if (index[9 + i / 8] & (1 << (i % 8))) != 0 {
                    count_before += 1;
                }
            }

            let mut offset = 9 + bitmap_bytes;
            for _ in 0..count_before {
                decode_varint(index, &mut offset)?;
            }

            decode_varint(index, &mut offset)
        }

        INDEX_EXACT_COMPRESSED => {
            if index.len() < 13 {
                return None;
            }

            let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
            let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

            if *query < min_val || *query > max_val {
                return Some(0);
            }

            let num_entries =
                u32::from_le_bytes([index[9], index[10], index[11], index[12]]) as usize;
            let mut offset = 13;

            let mut prev_key = 0i32;
            for i in 0..num_entries {
                if offset + 4 > index.len() {
                    return None;
                }

                let key = if i == 0 {
                    i32::from_le_bytes([
                        index[offset],
                        index[offset + 1],
                        index[offset + 2],
                        index[offset + 3],
                    ])
                } else {
                    let delta = i32::from_le_bytes([
                        index[offset],
                        index[offset + 1],
                        index[offset + 2],
                        index[offset + 3],
                    ]);
                    prev_key.wrapping_add(delta)
                };
                offset += 4;
                prev_key = key;

                let count = decode_varint(index, &mut offset)?;

                if key == *query {
                    return Some(count);
                }
            }

            Some(0)
        }

        INDEX_HYBRID => {
            if index.len() < 14 {
                return None;
            }

            let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
            let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

            if *query < min_val || *query > max_val {
                return Some(0);
            }

            let num_hashes = index[9];
            let num_top = u32::from_le_bytes([index[10], index[11], index[12], index[13]]) as usize;
            let mut offset = 14;

            let mut prev_key = 0i32;
            for i in 0..num_top {
                if offset + 4 > index.len() {
                    return None;
                }

                let key = if i == 0 {
                    i32::from_le_bytes([
                        index[offset],
                        index[offset + 1],
                        index[offset + 2],
                        index[offset + 3],
                    ])
                } else {
                    let delta = i32::from_le_bytes([
                        index[offset],
                        index[offset + 1],
                        index[offset + 2],
                        index[offset + 3],
                    ]);
                    prev_key.wrapping_add(delta)
                };
                offset += 4;
                prev_key = key;

                let count = decode_varint(index, &mut offset)?;

                if key == *query {
                    return Some(count);
                }
            }

            let bloom = &index[offset..];
            if bloom.is_empty() {
                return None;
            }

            let bloom_size = bloom.len();

            for i in 0..num_hashes {
                let hash = hash_value(*query, i as u32);
                let bit_pos = (hash as usize) % (bloom_size * 8);
                let byte_pos = bit_pos / 8;
                let bit_mask = 1 << (bit_pos % 8);

                if byte_pos >= bloom.len() || (bloom[byte_pos] & bit_mask) == 0 {
                    return Some(0);
                }
            }

            None
        }

        _ => None,
    }
}

fn hash_value(value: i32, seed: u32) -> u64 {
    let mut h = value as u64;
    h ^= seed as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccdu64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53u64);
    h ^= h >> 33;
    h
}
