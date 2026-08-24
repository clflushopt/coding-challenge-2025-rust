use crate::parameter::Parameter;
use std::collections::HashMap;

const BLOCK_SIZE: usize = 131072;

#[derive(Debug)]
enum IndexType {
    Empty,
    MinMax,
    Bitmap,
    HashMap,
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

    let index_type = if ratio < 0.1 {
        IndexType::MinMax
    } else if ratio < 0.5 {
        IndexType::MinMax
    } else if unique_count <= 1000 && ratio > 2.0 {
        IndexType::HashMap
    } else if unique_count < BLOCK_SIZE / 4 {
        let (min_val, max_val) = data.iter().fold((i32::MAX, i32::MIN), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
        let range = (max_val as i64 - min_val as i64 + 1) as usize;

        if range <= 100_000 && ratio > 1.0 {
            IndexType::Bitmap
        } else if unique_count <= 2000 {
            IndexType::HashMap
        } else {
            IndexType::MinMax
        }
    } else if unique_count <= 3000 && ratio > 1.5 {
        IndexType::HashMap
    } else {
        IndexType::MinMax
    };

    match index_type {
        IndexType::Empty => vec![0u8].into_boxed_slice(),
        IndexType::MinMax => build_minmax_index(data),
        IndexType::Bitmap => build_bitmap_index(data, &counts),
        IndexType::HashMap => build_hashmap_index(&counts),
    }
}

fn build_minmax_index(data: &[i32]) -> Box<[u8]> {
    let min_val = *data.iter().min().unwrap();
    let max_val = *data.iter().max().unwrap();

    let mut result = Vec::new();
    result.push(3u8);
    result.extend_from_slice(&min_val.to_le_bytes());
    result.extend_from_slice(&max_val.to_le_bytes());

    result.into_boxed_slice()
}

fn build_bitmap_index(data: &[i32], counts: &HashMap<i32, u64>) -> Box<[u8]> {
    let min_val = *data.iter().min().unwrap();
    let max_val = *data.iter().max().unwrap();
    let range = (max_val as i64 - min_val as i64 + 1) as usize;

    let mut result = Vec::new();
    result.push(1u8);

    result.extend_from_slice(&min_val.to_le_bytes());
    result.extend_from_slice(&max_val.to_le_bytes());

    let bitmap_bytes = (range + 7) / 8;
    let mut bitmap = vec![0u8; bitmap_bytes];
    let mut count_map: Vec<(i32, u64)> = Vec::new();

    for (&val, &count) in counts {
        let offset = (val - min_val) as usize;
        bitmap[offset / 8] |= 1u8 << (offset % 8);
        count_map.push((val, count));
    }

    result.extend_from_slice(&bitmap);

    result.extend_from_slice(&(count_map.len() as u32).to_le_bytes());
    for (val, count) in count_map {
        result.extend_from_slice(&val.to_le_bytes());
        result.extend_from_slice(&count.to_le_bytes());
    }

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

pub fn query_idx(_parameter: &Parameter, index: &[u8], query: &i32) -> Option<u64> {
    if index.is_empty() {
        return None;
    }

    let index_type = index[0];

    match index_type {
        0 => None,
        1 => query_bitmap_index(index, query),
        2 => query_hashmap_index(index, query),
        3 => query_minmax_index(index, query),
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

fn query_bitmap_index(index: &[u8], query: &i32) -> Option<u64> {
    if index.len() < 9 {
        return None;
    }

    let min_val = i32::from_le_bytes([index[1], index[2], index[3], index[4]]);
    let max_val = i32::from_le_bytes([index[5], index[6], index[7], index[8]]);

    if *query < min_val || *query > max_val {
        return Some(0);
    }

    let offset = (*query - min_val) as usize;
    let range = (max_val as i64 - min_val as i64 + 1) as usize;
    let bitmap_bytes = (range + 7) / 8;

    if index.len() < 9 + bitmap_bytes + 4 {
        return None;
    }

    let bitmap = &index[9..9 + bitmap_bytes];
    let bit_set = (bitmap[offset / 8] & (1u8 << (offset % 8))) != 0;

    if !bit_set {
        return Some(0);
    }

    let count_map_start = 9 + bitmap_bytes;
    let count_map_len = u32::from_le_bytes([
        index[count_map_start],
        index[count_map_start + 1],
        index[count_map_start + 2],
        index[count_map_start + 3],
    ]) as usize;

    let mut pos = count_map_start + 4;
    for _ in 0..count_map_len {
        if pos + 12 > index.len() {
            return None;
        }

        let val = i32::from_le_bytes([index[pos], index[pos + 1], index[pos + 2], index[pos + 3]]);
        let count = u64::from_le_bytes([
            index[pos + 4],
            index[pos + 5],
            index[pos + 6],
            index[pos + 7],
            index[pos + 8],
            index[pos + 9],
            index[pos + 10],
            index[pos + 11],
        ]);

        if val == *query {
            return Some(count);
        }

        pos += 12;
    }

    None
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
