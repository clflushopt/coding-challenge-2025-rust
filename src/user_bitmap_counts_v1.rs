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
    let min_val = *data.iter().min().unwrap();
    let max_val = *data.iter().max().unwrap();
    let range = (max_val as i64 - min_val as i64 + 1).max(0) as u64;

    let hashmap_size_kb = (5.0 + unique_count as f64 * 12.0) / 1024.0;
    let bitmap_size_kb = (9.0 + (range as f64 / 8.0)) / 1024.0;
    let hybrid_size_kb = bitmap_size_kb + (unique_count as f64 * 12.0) / 1024.0;

    if unique_count <= 100 {
        return build_hashmap_index(&counts);
    }

    if unique_count <= 500 && ratio > 0.4 {
        return build_hashmap_index(&counts);
    }

    if unique_count <= 1000 && ratio > hashmap_size_kb * 0.6 {
        return build_hashmap_index(&counts);
    }

    if range < 100000 && unique_count > 500 && ratio > 2.0 {
        if hybrid_size_kb < 25.0 && hybrid_size_kb < hashmap_size_kb * 1.3 {
            return build_hybrid_index(min_val, max_val, &counts);
        }
    }

    if range < 150000 && bitmap_size_kb < 20.0 && ratio > 0.6 {
        if bitmap_size_kb < hashmap_size_kb * 0.7 || (range < 50000 && ratio > 1.0) {
            return build_bitmap_index(min_val, max_val, &counts);
        }
    }

    if unique_count <= 2000 && ratio > hashmap_size_kb * 1.0 {
        return build_hashmap_index(&counts);
    }

    if unique_count <= 5000 && ratio > hashmap_size_kb * 2.0 {
        return build_hashmap_index(&counts);
    }

    build_minmax_index(min_val, max_val)
}

fn build_minmax_index(min_val: i32, max_val: i32) -> Box<[u8]> {
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

fn build_bitmap_index(min_val: i32, max_val: i32, counts: &HashMap<i32, u64>) -> Box<[u8]> {
    let mut result = Vec::new();
    result.push(3u8);
    result.extend_from_slice(&min_val.to_le_bytes());
    result.extend_from_slice(&max_val.to_le_bytes());

    let range = (max_val as i64 - min_val as i64 + 1) as usize;
    let bitmap_bytes = (range + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_bytes];

    for &val in counts.keys() {
        let offset = (val as i64 - min_val as i64) as usize;
        bitmap[offset / 8] |= 1 << (offset % 8);
    }

    result.extend_from_slice(&bitmap);
    result.into_boxed_slice()
}

fn build_hybrid_index(min_val: i32, max_val: i32, counts: &HashMap<i32, u64>) -> Box<[u8]> {
    let mut result = Vec::new();
    result.push(4u8);
    result.extend_from_slice(&min_val.to_le_bytes());
    result.extend_from_slice(&max_val.to_le_bytes());

    let range = (max_val as i64 - min_val as i64 + 1) as usize;
    let bitmap_bytes = (range + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_bytes];

    for &val in counts.keys() {
        let offset = (val as i64 - min_val as i64) as usize;
        bitmap[offset / 8] |= 1 << (offset % 8);
    }

    result.extend_from_slice(&bitmap);

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

    match index[0] {
        1 => query_minmax_index(index, query),
        2 => query_hashmap_index(index, query),
        3 => query_bitmap_index(index, query),
        4 => query_hybrid_index(index, query),
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

fn query_bitmap_index(index: &[u8], query: &i32) -> Option<u64> {
    if index.len() < 9 {
        return None;
    }

    let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
    let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

    if *query < min_val || *query > max_val {
        return Some(0);
    }

    let offset = (*query as i64 - min_val as i64) as usize;
    let byte_idx = 9 + offset / 8;

    if byte_idx >= index.len() {
        return None;
    }

    if (index[byte_idx] & (1 << (offset % 8))) != 0 {
        None
    } else {
        Some(0)
    }
}

fn query_hybrid_index(index: &[u8], query: &i32) -> Option<u64> {
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
    let offset = (*query as i64 - min_val as i64) as usize;
    let byte_idx = 9 + offset / 8;

    if byte_idx >= 9 + bitmap_bytes {
        return None;
    }

    if (index[byte_idx] & (1 << (offset % 8))) == 0 {
        return Some(0);
    }

    let count_table_start = 9 + bitmap_bytes;
    if count_table_start + 4 > index.len() {
        return None;
    }

    let count = u32::from_le_bytes([
        index[count_table_start],
        index[count_table_start + 1],
        index[count_table_start + 2],
        index[count_table_start + 3],
    ]) as usize;

    let entry_size = 12;
    let mut left = 0;
    let mut right = count;
    let table_start = count_table_start + 4;

    while left < right {
        let mid = (left + right) / 2;
        let pos = table_start + mid * entry_size;

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

    None
}
