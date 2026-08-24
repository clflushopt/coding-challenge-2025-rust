use crate::parameter::Parameter;
use std::collections::HashMap;


const INDEX_EMPTY: u8 = 0;
const INDEX_EXACT: u8 = 1;
const INDEX_BLOOM_TOPK: u8 = 2;

pub fn build_idx(parameter: &Parameter, data: &[i32]) -> Box<[u8]> {
    if data.is_empty() {
        return vec![INDEX_EMPTY].into_boxed_slice();
    }

    let mut freq_map: HashMap<i32, u64> = HashMap::new();
    for &val in data {
        *freq_map.entry(val).or_insert(0) += 1;
    }

    let distinct_count = freq_map.len();

    let exact_size = 1 + 4 + (distinct_count * 12);

    let ratio = parameter.factor_skip as f64 / parameter.factor_size as f64;

    if distinct_count <= 10 || (distinct_count <= 100 && ratio > 5.0) {
        return build_exact_index(&freq_map);
    }

    if distinct_count <= 1000 && ratio > 2.0 {
        return build_exact_index(&freq_map);
    }

    let top_k = if ratio > 3.0 {
        50
    } else if ratio > 1.0 {
        20
    } else {
        10
    };

    build_bloom_topk_index(&freq_map, data, top_k)
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

fn build_bloom_topk_index(freq_map: &HashMap<i32, u64>, data: &[i32], top_k: usize) -> Box<[u8]> {
    let mut sorted_freq: Vec<_> = freq_map.iter().collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(a.1));

    let top_entries: Vec<_> = sorted_freq
        .iter()
        .take(top_k)
        .map(|(&k, &v)| (k, v))
        .collect();

    let bloom_size = 2048;
    let num_hashes = 7;
    let mut bloom = vec![0u8; bloom_size];

    for &val in data {
        if top_entries.iter().any(|(k, _)| *k == val) {
            continue;
        }

        for i in 0..num_hashes {
            let hash = hash_value(val, i);
            let bit_pos = (hash as usize) % (bloom_size * 8);
            bloom[bit_pos / 8] |= 1 << (bit_pos % 8);
        }
    }

    let mut buffer = Vec::new();
    buffer.push(INDEX_BLOOM_TOPK);
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

                if key == *query {
                    return Some(count);
                }

                offset += 12;
            }

            Some(0)
        }

        INDEX_BLOOM_TOPK => {
            if index.len() < 5 {
                return None;
            }

            let num_top = u32::from_le_bytes([index[1], index[2], index[3], index[4]]) as usize;
            let mut offset = 5;

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

                if key == *query {
                    return Some(count);
                }

                offset += 12;
            }

            let bloom = &index[offset..];
            if bloom.is_empty() {
                return None;
            }

            let bloom_size = bloom.len();
            let num_hashes = 7;

            let mut all_present = true;
            for i in 0..num_hashes {
                let hash = hash_value(*query, i);
                let bit_pos = (hash as usize) % (bloom_size * 8);
                let byte_pos = bit_pos / 8;
                let bit_mask = 1 << (bit_pos % 8);

                if byte_pos >= bloom.len() || (bloom[byte_pos] & bit_mask) == 0 {
                    all_present = false;
                    break;
                }
            }

            if !all_present {
                Some(0)
            } else {
                None
            }
        }

        _ => None,
    }
}

fn hash_value(value: i32, seed: u32) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash ^= seed as u64;

    let bytes = value.to_le_bytes();
    for &byte in &bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }

    hash
}
