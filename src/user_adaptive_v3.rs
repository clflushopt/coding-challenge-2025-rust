use crate::parameter::Parameter;
use std::collections::HashMap;

const BLOCK_SIZE: usize = 131072;

pub fn build_idx(parameter: &Parameter, data: &[i32]) -> Box<[u8]> {
    let fa = parameter.factor_skip as f64;
    let fs = parameter.factor_size as f64;
    let ratio = fa / fs;

    let mut counts: HashMap<i32, u64> = HashMap::new();
    for &val in data {
        *counts.entry(val).or_insert(0) += 1;
    }

    let unique_count = counts.len();

    let hashmap_kb = (5.0 + unique_count as f64 * 12.0) / 1024.0;



    let use_hashmap = if unique_count <= 50 {
        ratio > 0.5
    } else if unique_count <= 200 {
        ratio > 1.0
    } else if unique_count <= 500 {
        ratio > 2.0
    } else if unique_count <= 1000 {
        ratio > 4.0
    } else if unique_count <= 2000 {
        ratio > 8.0
    } else if unique_count <= 5000 {
        ratio > 20.0
    } else {
        false
    };

    if use_hashmap {
        build_hashmap_index(&counts)
    } else {
        build_minmax_index(data)
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

fn build_hashmap_index(counts: &HashMap<i32, u64>) -> Box<[u8]> {
    let mut result = Vec::new();
    result.push(2u8);

    result.extend_from_slice(&(counts.len() as u32).to_le_bytes());

    let mut entries: Vec<_> = counts.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_by_key(|&(k, _)| k);

    for (val, count) in entries {
        result.extend_from_slice(&val.to_le_bytes());
        result.extend_from_slice(&count.to_le_bytes());
    }

    result.into_boxed_slice()
}

pub fn query_idx(parameter: &Parameter, index: &[u8], query: &i32) -> Option<u64> {
    if index.is_empty() {
        return None;
    }

    let index_type = index[0];

    match index_type {
        1 => query_minmax_index(index, query),
        2 => query_hashmap_index(index, query),
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

fn query_hashmap_index(index: &[u8], query: &i32) -> Option<u64> {
    if index.len() < 5 {
        return None;
    }

    let count = u32::from_le_bytes([index[1], index[2], index[3], index[4]]) as usize;

    let entry_size = 12;
    let mut left = 0;
    let mut right = count;

    while left < right {
        let mid = (left + right) / 2;
        let pos = 5 + mid * entry_size;

        if pos + entry_size > index.len() {
            return None;
        }

        let val = i32::from_le_bytes([index[pos], index[pos + 1], index[pos + 2], index[pos + 3]]);

        if val == *query {
            let cnt = u64::from_le_bytes([
                index[pos + 4],
                index[pos + 5],
                index[pos + 6],
                index[pos + 7],
                index[pos + 8],
                index[pos + 9],
                index[pos + 10],
                index[pos + 11],
            ]);
            return Some(cnt);
        } else if val < *query {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    Some(0)
}
