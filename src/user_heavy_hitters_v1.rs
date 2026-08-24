use crate::parameter::Parameter;
use std::collections::HashMap;

const INDEX_EMPTY: u8 = 0;
const INDEX_EXACT_COMPRESSED: u8 = 1;
const INDEX_HYBRID: u8 = 2;
const INDEX_HEAVY_HITTER: u8 = 3;

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

    let mut sorted_freq: Vec<_> = freq_map.iter().map(|(k, v)| (*k, *v)).collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(&a.1));

    let mut coverage_20 = 0u64;
    let mut coverage_50 = 0u64;
    let mut coverage_100 = 0u64;

    for (i, (_, count)) in sorted_freq.iter().enumerate() {
        if i < 20 {
            coverage_20 += count;
        }
        if i < 50 {
            coverage_50 += count;
        }
        if i < 100 {
            coverage_100 += count;
        }
    }

    let coverage_20_pct = coverage_20 as f64 / total_elements as f64;
    let coverage_50_pct = coverage_50 as f64 / total_elements as f64;
    let coverage_100_pct = coverage_100 as f64 / total_elements as f64;

    if distinct_count <= 120 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    if distinct_count <= 350 && ratio > 12.0 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    if distinct_count <= 600 && ratio > 22.0 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    if distinct_count <= 1000 && ratio > 40.0 {
        return build_exact_compressed(&freq_map, min_val, max_val);
    }

    if ratio > 15.0 && coverage_20_pct > 0.7 {
        return build_heavy_hitter(&freq_map, 20, min_val, max_val);
    }

    if ratio > 10.0 && coverage_50_pct > 0.75 {
        return build_heavy_hitter(&freq_map, 50, min_val, max_val);
    }

    if ratio > 7.0 && coverage_100_pct > 0.80 {
        return build_heavy_hitter(&freq_map, 100, min_val, max_val);
    }

    let top_k = calculate_top_k(ratio, distinct_count, coverage_50_pct);
    let bloom_size = calculate_bloom_size(ratio, distinct_count, coverage_50_pct);

    build_hybrid_index(&freq_map, data, top_k, bloom_size, min_val, max_val)
}

fn calculate_top_k(ratio: f64, distinct_count: usize, coverage: f64) -> usize {
    let base_k = if ratio > 35.0 {
        if coverage > 0.5 {
            250
        } else {
            180
        }
    } else if ratio > 25.0 {
        if coverage > 0.5 {
            200
        } else {
            140
        }
    } else if ratio > 15.0 {
        if coverage > 0.4 {
            150
        } else {
            100
        }
    } else if ratio > 10.0 {
        if coverage > 0.4 {
            110
        } else {
            75
        }
    } else if ratio > 6.0 {
        if coverage > 0.3 {
            70
        } else {
            50
        }
    } else if ratio > 3.0 {
        if coverage > 0.25 {
            40
        } else {
            25
        }
    } else {
        15
    };

    base_k.min(distinct_count / 3).max(10)
}

fn calculate_bloom_size(ratio: f64, distinct_count: usize, coverage: f64) -> usize {
    let target_fpr = if coverage > 0.6 {
        0.025
    } else if coverage > 0.4 {
        0.015
    } else if ratio > 30.0 {
        0.002
    } else if ratio > 20.0 {
        0.004
    } else if ratio > 12.0 {
        0.008
    } else if ratio > 6.0 {
        0.018
    } else if ratio > 3.0 {
        0.035
    } else {
        0.070
    };

    let ln2_squared = 0.4804530139182014;
    let optimal_bits =
        (-1.0 * distinct_count as f64 * (target_fpr as f64).ln() / ln2_squared).ceil() as usize;
    let optimal_bytes = (optimal_bits + 7) / 8;

    let (min_size, max_size) = if ratio > 30.0 {
        (1024, 5120)
    } else if ratio > 20.0 {
        (768, 4096)
    } else if ratio > 12.0 {
        (640, 3072)
    } else if ratio > 6.0 {
        (512, 2048)
    } else if ratio > 3.0 {
        (384, 1280)
    } else {
        (256, 896)
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

fn build_heavy_hitter(
    freq_map: &HashMap<i32, u64>,
    top_n: usize,
    min_val: i32,
    max_val: i32,
) -> Box<[u8]> {
    let mut sorted_freq: Vec<_> = freq_map.iter().collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(a.1));

    let mut top_entries: Vec<_> = sorted_freq
        .iter()
        .take(top_n.min(freq_map.len()))
        .map(|(&k, &v)| (k, v))
        .collect();

    top_entries.sort_unstable_by_key(|&(k, _)| k);

    let mut buffer = Vec::new();
    buffer.push(INDEX_HEAVY_HITTER);

    buffer.extend_from_slice(&min_val.to_le_bytes());
    buffer.extend_from_slice(&max_val.to_le_bytes());
    buffer.extend_from_slice(&(top_entries.len() as u16).to_le_bytes());

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

    let num_hashes = optimal_k.clamp(3, 5);

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

    match index[0] {
        INDEX_EMPTY => None,

        INDEX_HEAVY_HITTER => {
            if index.len() < 11 {
                return None;
            }

            let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
            let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

            if *query < min_val || *query > max_val {
                return Some(0);
            }

            let num_entries = u16::from_le_bytes([index[9], index[10]]) as usize;
            let mut offset = 11;

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

            None
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
