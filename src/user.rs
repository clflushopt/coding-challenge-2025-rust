use crate::parameter::Parameter;
use std::collections::HashMap;

const BLOCK_SIZE: usize = 131072;

fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        result.push(byte);
        if value == 0 {
            break;
        }
    }
    result
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

fn zigzag_encode(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

fn zigzag_decode(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

pub fn build_idx(parameter: &Parameter, data: &[i32]) -> Box<[u8]> {
    let fa = parameter.factor_skip as f64;
    let fs = parameter.factor_size as f64;
    let ratio = fa / fs;

    let mut counts: HashMap<i32, u64> = HashMap::new();
    for &val in data {
        *counts.entry(val).or_insert(0) += 1;
    }

    let unique_count = counts.len();

    let minmax_index = build_minmax_index(data);

    if unique_count > 5000 || ratio < 0.3 {
        return minmax_index;
    }

    let hashmap_index = build_compressed_hashmap(&counts);
    let hashmap_kb = hashmap_index.len() as f64 / 1024.0;

    let topk_index = if unique_count > 500 {
        Some(build_topk_index(&counts, ratio))
    } else {
        None
    };

    let use_hashmap = if unique_count <= 50 {
        ratio > 0.4
    } else if unique_count <= 100 {
        ratio > 0.8
    } else if unique_count <= 200 {
        ratio > 1.2
    } else if unique_count <= 500 {
        ratio > 2.5
    } else if unique_count <= 1000 {
        ratio > 5.0
    } else if unique_count <= 2000 {
        ratio > 10.0
    } else {
        ratio > 25.0
    };

    if use_hashmap && hashmap_kb < 10.0 {
        hashmap_index
    } else if let Some(topk) = topk_index {
        let topk_kb = topk.len() as f64 / 1024.0;
        if topk_kb < 3.0 && ratio > 1.0 {
            topk
        } else {
            minmax_index
        }
    } else {
        minmax_index
    }
}

fn build_minmax_index(data: &[i32]) -> Box<[u8]> {
    let min_val = *data.iter().min().unwrap();
    let max_val = *data.iter().max().unwrap();

    let mut result = Vec::new();
    result.push(1u8);
    result.extend_from_slice(&min_val.to_le_bytes());
    result.extend_from_slice(&max_val.to_le_bytes());

    result.into_boxed_slice()
}

fn build_compressed_hashmap(counts: &HashMap<i32, u64>) -> Box<[u8]> {
    let mut result = Vec::new();
    result.push(2u8);

    let mut entries: Vec<_> = counts.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_by_key(|&(k, _)| k);

    let count_bytes = encode_varint(entries.len() as u64);
    result.extend_from_slice(&count_bytes);

    let mut prev_key = 0i32;
    for (key, count) in entries {
        let delta = key.wrapping_sub(prev_key);
        let encoded_delta = zigzag_encode(delta);
        result.extend_from_slice(&encode_varint(encoded_delta as u64));
        result.extend_from_slice(&encode_varint(count));
        prev_key = key;
    }

    result.into_boxed_slice()
}

fn build_topk_index(counts: &HashMap<i32, u64>, ratio: f64) -> Box<[u8]> {
    let k = if ratio > 5.0 {
        200
    } else if ratio > 2.0 {
        100
    } else {
        50
    };

    let mut entries: Vec<_> = counts.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.truncate(k.min(entries.len()));
    entries.sort_by_key(|&(k, _)| k);

    let mut result = Vec::new();
    result.push(3u8);

    let count_bytes = encode_varint(entries.len() as u64);
    result.extend_from_slice(&count_bytes);

    let mut prev_key = 0i32;
    for (key, count) in entries {
        let delta = key.wrapping_sub(prev_key);
        let encoded_delta = zigzag_encode(delta);
        result.extend_from_slice(&encode_varint(encoded_delta as u64));
        result.extend_from_slice(&encode_varint(count));
        prev_key = key;
    }

    result.into_boxed_slice()
}

pub fn query_idx(parameter: &Parameter, index: &[u8], query: &i32) -> Option<u64> {
    if index.is_empty() {
        return None;
    }

    match index[0] {
        1 => query_minmax_index(index, query),
        2 => query_compressed_hashmap(index, query),
        3 => query_topk_index(index, query),
        _ => None,
    }
}

fn query_minmax_index(index: &[u8], query: &i32) -> Option<u64> {
    if index.len() < 9 {
        return None;
    }

    let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
    let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

    if *query < min_val || *query > max_val {
        Some(0)
    } else {
        None
    }
}

fn query_compressed_hashmap(index: &[u8], query: &i32) -> Option<u64> {
    let mut offset = 1;
    let count = decode_varint(index, &mut offset)? as usize;

    let mut left = 0;
    let mut right = count;

    let mut entries = Vec::with_capacity(count);
    let mut current_key = 0i32;
    let mut temp_offset = offset;

    for _ in 0..count {
        let delta = decode_varint(index, &mut temp_offset)? as u32;
        let delta_signed = zigzag_decode(delta);
        current_key = current_key.wrapping_add(delta_signed);
        let cnt = decode_varint(index, &mut temp_offset)?;
        entries.push((current_key, cnt));
    }

    while left < right {
        let mid = (left + right) / 2;
        if entries[mid].0 == *query {
            return Some(entries[mid].1);
        } else if entries[mid].0 < *query {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    Some(0)
}

fn query_topk_index(index: &[u8], query: &i32) -> Option<u64> {
    let mut offset = 1;
    let count = decode_varint(index, &mut offset)? as usize;

    let mut entries = Vec::with_capacity(count);
    let mut current_key = 0i32;

    for _ in 0..count {
        let delta = decode_varint(index, &mut offset)? as u32;
        let delta_signed = zigzag_decode(delta);
        current_key = current_key.wrapping_add(delta_signed);
        let cnt = decode_varint(index, &mut offset)?;
        entries.push((current_key, cnt));
    }

    let mut left = 0;
    let mut right = count;

    while left < right {
        let mid = (left + right) / 2;
        if entries[mid].0 == *query {
            return Some(entries[mid].1);
        } else if entries[mid].0 < *query {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    None
}
