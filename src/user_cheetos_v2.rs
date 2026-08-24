use crate::parameter::Parameter;
use std::collections::HashMap;

const INDEX_EMPTY: u8 = 0;
const INDEX_EXACT_COMPACT: u8 = 1;
const INDEX_BLOOM_ONLY: u8 = 2;
const INDEX_BLOOM_TOPK: u8 = 3;
const INDEX_MINMAX: u8 = 4;

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

    if distinct_count <= 5 {
        return build_exact_index(&freq_map);
    }

    if distinct_count <= 50 && ratio > 10.0 {
        return build_exact_index(&freq_map);
    }

    if distinct_count <= 200 && ratio > 20.0 {
        return build_exact_index(&freq_map);
    }

    if ratio < 0.5 {
        return build_minmax_index(data);
    }

    if distinct_count <= 5000 && ratio < 2.0 {
        return build_bloom_only(data, calculate_optimal_bloom_size(distinct_count, ratio));
    }

    let (top_k, bloom_size) =
        calculate_adaptive_params(parameter, distinct_count, ratio, &freq_map);
    build_bloom_topk_index(&freq_map, data, top_k, bloom_size)
}

fn calculate_optimal_bloom_size(distinct_count: usize, ratio: f64) -> usize {

    let target_fpr = if ratio > 5.0 {
        0.001
    } else if ratio > 2.0 {
        0.01
    } else {
        0.05
    };

    let optimal_bits = (-1.0 * distinct_count as f64 * (target_fpr as f64).ln()
        / (2.0_f64.ln().powi(2)))
    .ceil() as usize;
    let optimal_bytes = (optimal_bits + 7) / 8;

    optimal_bytes.max(512).min(8192)
}

fn calculate_adaptive_params(
    parameter: &Parameter,
    distinct_count: usize,
    ratio: f64,
    freq_map: &HashMap<i32, u64>,
) -> (usize, usize) {
    let total_count: u64 = freq_map.values().sum();

    let mut sorted_freq: Vec<_> = freq_map.values().copied().collect();
    sorted_freq.sort_by(|a, b| b.cmp(a));

    let mut best_k = 10;
    let mut cumulative = 0u64;

    for (k, &freq) in sorted_freq.iter().enumerate() {
        cumulative += freq;
        let coverage = cumulative as f64 / total_count as f64;

        let storage_cost = (k + 1) as f64 * 12.0 / 1024.0;
        let expected_benefit = coverage * ratio;

        if expected_benefit > storage_cost * parameter.factor_size as f64 {
            best_k = k + 1;
        } else {
            break;
        }

        if k >= 200 {
            break;
        }
    }

    best_k = best_k.max(5).min(100);

    let remaining_distinct = distinct_count.saturating_sub(best_k);
    let bloom_size = calculate_optimal_bloom_size(remaining_distinct, ratio);

    (best_k, bloom_size)
}

fn build_minmax_index(data: &[i32]) -> Box<[u8]> {
    let min = *data.iter().min().unwrap();
    let max = *data.iter().max().unwrap();

    let mut buffer = Vec::with_capacity(9);
    buffer.push(INDEX_MINMAX);
    buffer.extend_from_slice(&min.to_le_bytes());
    buffer.extend_from_slice(&max.to_le_bytes());
    buffer.into_boxed_slice()
}

fn build_exact_index(freq_map: &HashMap<i32, u64>) -> Box<[u8]> {
    let num_entries = freq_map.len() as u32;
    let size = 1 + 4 + (num_entries as usize * 12);
    let mut buffer = Vec::with_capacity(size);

    buffer.push(INDEX_EXACT_COMPACT);
    buffer.extend_from_slice(&num_entries.to_le_bytes());

    for (&key, &count) in freq_map.iter() {
        buffer.extend_from_slice(&key.to_le_bytes());
        buffer.extend_from_slice(&count.to_le_bytes());
    }

    buffer.into_boxed_slice()
}

fn build_bloom_only(data: &[i32], bloom_size: usize) -> Box<[u8]> {
    let num_hashes = calculate_optimal_hash_count(bloom_size, data.len());
    let mut bloom = vec![0u8; bloom_size];

    for &val in data {
        for i in 0..num_hashes {
            let hash = hash_value(val, i);
            let bit_pos = (hash as usize) % (bloom_size * 8);
            bloom[bit_pos / 8] |= 1 << (bit_pos % 8);
        }
    }

    let mut buffer = Vec::with_capacity(2 + bloom_size);
    buffer.push(INDEX_BLOOM_ONLY);
    buffer.push(num_hashes as u8);
    buffer.extend_from_slice(&bloom);
    buffer.into_boxed_slice()
}

fn build_bloom_topk_index(
    freq_map: &HashMap<i32, u64>,
    data: &[i32],
    top_k: usize,
    bloom_size: usize,
) -> Box<[u8]> {
    let mut sorted_freq: Vec<_> = freq_map.iter().collect();
    sorted_freq.sort_by(|a, b| b.1.cmp(a.1));

    let top_entries: Vec<_> = sorted_freq
        .iter()
        .take(top_k)
        .map(|(&k, &v)| (k, v))
        .collect();
    let top_keys: Vec<i32> = top_entries.iter().map(|(k, _)| *k).collect();

    let num_hashes = calculate_optimal_hash_count(
        bloom_size,
        data.len() - top_entries.iter().map(|(_, c)| c).sum::<u64>() as usize,
    );
    let mut bloom = vec![0u8; bloom_size];

    for &val in data {
        if top_keys.contains(&val) {
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
    buffer.push(num_hashes as u8);
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

        INDEX_MINMAX => {
            if index.len() < 9 {
                return None;
            }

            let min = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
            let max = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

            if *query < min || *query > max {
                Some(0)
            } else {
                None
            }
        }

        INDEX_EXACT_COMPACT => {
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

        INDEX_BLOOM_ONLY => {
            if index.len() < 2 {
                return None;
            }

            let num_hashes = index[1] as u32;
            let bloom = &index[2..];

            if bloom.is_empty() {
                return None;
            }

            if check_bloom(bloom, *query, num_hashes) {
                None
            } else {
                Some(0)
            }
        }

        INDEX_BLOOM_TOPK => {
            if index.len() < 6 {
                return None;
            }

            let num_hashes = index[1] as u32;
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

            if check_bloom(bloom, *query, num_hashes) {
                None
            } else {
                Some(0)
            }
        }

        _ => None,
    }
}

fn check_bloom(bloom: &[u8], value: i32, num_hashes: u32) -> bool {
    let bloom_size = bloom.len();

    for i in 0..num_hashes {
        let hash = hash_value(value, i);
        let bit_pos = (hash as usize) % (bloom_size * 8);
        let byte_pos = bit_pos / 8;
        let bit_mask = 1 << (bit_pos % 8);

        if byte_pos >= bloom.len() || (bloom[byte_pos] & bit_mask) == 0 {
            return false;
        }
    }

    true
}

fn calculate_optimal_hash_count(bloom_size_bytes: usize, num_elements: usize) -> u32 {
    if num_elements == 0 {
        return 7;
    }

    let m = bloom_size_bytes * 8;
    let n = num_elements;
    let optimal = ((m as f64 / n as f64) * 2.0_f64.ln()).ceil() as u32;

    optimal.max(3).min(15)
}

fn hash_value(value: i32, seed: u32) -> u64 {
    let mut x = value as u64;
    x ^= seed as u64;
    x = x.wrapping_mul(0x9E3779B97F4A7C15u64);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9u64);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EBu64);
    x ^= x >> 31;
    x
}
