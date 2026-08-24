use crate::parameter::Parameter;
use std::collections::HashMap;

const INDEX_EMPTY: u8 = 0;
const INDEX_EXACT: u8 = 1;
const INDEX_HYBRID: u8 = 2;

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

    if distinct_count <= 100 {
        return build_exact_index(&freq_map);
    }

    if distinct_count <= 500 && ratio > 5.0 {
        return build_exact_index(&freq_map);
    }

    if distinct_count <= 1500 && ratio > 15.0 {
        return build_exact_index(&freq_map);
    }

    let top_k = if ratio > 20.0 {
        100
    } else if ratio > 10.0 {
        50
    } else if ratio > 5.0 {
        30
    } else if ratio > 2.0 {
        15
    } else {
        5
    };

    let bloom_size = if ratio > 10.0 {
        8192
    } else if ratio > 5.0 {
        4096
    } else if ratio > 2.0 {
        2048
    } else {
        1024
    };

    build_hybrid_index(&freq_map, data, top_k, bloom_size)
}

fn build_exact_index(freq_map: &HashMap<i32, u64>) -> Box<[u8]> {
    let num_entries = freq_map.len() as u32;
    let size = 1 + 4 + (num_entries as usize * 12);
    let mut buffer = Vec::with_capacity(size);

    buffer.push(INDEX_EXACT);
    buffer.extend_from_slice(&num_entries.to_le_bytes());

    for (&key, &count) in freq_map.iter() {
        buffer.extend_from_slice(&key.to_le_bytes());
        buffer.extend_from_slice(&count.to_le_bytes());
    }

    buffer.into_boxed_slice()
}

fn build_hybrid_index(
    freq_map: &HashMap<i32, u64>,
    data: &[i32],
    top_k: usize,
    bloom_size: usize,
) -> Box<[u8]> {
    let mut sorted_freq: Vec<_> = freq_map.iter().collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(a.1));

    let top_entries: Vec<_> = sorted_freq
        .iter()
        .take(top_k.min(freq_map.len()))
        .map(|(&k, &v)| (k, v))
        .collect();

    let top_keys: std::collections::HashSet<i32> = top_entries.iter().map(|(k, _)| *k).collect();

    let num_hashes = 10u8;
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
    buffer.push(num_hashes);
    buffer.extend_from_slice(&(top_entries.len() as u32).to_le_bytes());

    for (key, count) in top_entries {
        buffer.extend_from_slice(&key.to_le_bytes());
        buffer.extend_from_slice(&count.to_le_bytes());
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

        INDEX_EXACT => {
            if index.len() < 5 {
                return None;
            }

            let num_entries = u32::from_le_bytes([index[1], index[2], index[3], index[4]]) as usize;
            let mut offset = 5;

            for _ in 0..num_entries {
                if offset + 12 > index.len() {
                    return None;
                }

                let key = i32::from_le_bytes([
                    index[offset],
                    index[offset + 1],
                    index[offset + 2],
                    index[offset + 3],
                ]);

                if key == *query {
                    let count = u64::from_le_bytes([
                        index[offset + 4],
                        index[offset + 5],
                        index[offset + 6],
                        index[offset + 7],
                        index[offset + 8],
                        index[offset + 9],
                        index[offset + 10],
                        index[offset + 11],
                    ]);
                    return Some(count);
                }

                offset += 12;
            }

            Some(0)
        }

        INDEX_HYBRID => {
            if index.len() < 6 {
                return None;
            }

            let num_hashes = index[1];
            let num_top = u32::from_le_bytes([index[2], index[3], index[4], index[5]]) as usize;
            let mut offset = 6;

            for _ in 0..num_top {
                if offset + 12 > index.len() {
                    return None;
                }

                let key = i32::from_le_bytes([
                    index[offset],
                    index[offset + 1],
                    index[offset + 2],
                    index[offset + 3],
                ]);

                if key == *query {
                    let count = u64::from_le_bytes([
                        index[offset + 4],
                        index[offset + 5],
                        index[offset + 6],
                        index[offset + 7],
                        index[offset + 8],
                        index[offset + 9],
                        index[offset + 10],
                        index[offset + 11],
                    ]);
                    return Some(count);
                }

                offset += 12;
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
