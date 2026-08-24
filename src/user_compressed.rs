use crate::parameter::Parameter;
use std::collections::HashMap;

fn encode_u32(mut x: u32, out: &mut Vec<u8>) {
    while x >= 0x80 {
        out.push(((x & 0x7F) as u8) | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

fn decode_u32(data: &[u8], pos: &mut usize) -> Option<u32> {
    let mut result: u32 = 0;
    let mut shift = 0;
    while *pos < data.len() {
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 32 {
            return None;
        }
    }
    None
}

fn encode_i32_zigzag(x: i32) -> u32 {
    ((x << 1) ^ (x >> 31)) as u32
}
fn decode_i32_zigzag(x: u32) -> i32 {
    ((x >> 1) as i32) ^ (-((x & 1) as i32))
}

#[allow(unused_variables)]
pub fn build_idx(parameter: &Parameter, data: &[i32]) -> Box<[u8]> {
    let mut freq: HashMap<i32, u32> = HashMap::with_capacity(4096);
    for &v in data {
        *freq.entry(v).or_insert(0) += 1;
    }

    let n_rows = data.len() as f64;
    let distinct = freq.len();
    let fs = (parameter.factor_size.max(1)) as f64;
    let fa = (parameter.factor_skip.max(1)) as f64;

    let entry_bytes = 3.0_f64;
    let threshold_f = n_rows * (fs / fa) * (entry_bytes / 1024.0);
    let mut threshold = threshold_f.ceil() as u32;
    if threshold < 1 {
        threshold = 1;
    }

    if distinct <= 4096 {
        threshold = 1;
    }

    let mut keep: Vec<(i32, u32)> = freq
        .into_iter()
        .filter(|&(_, cnt)| cnt >= threshold)
        .collect();

    keep.sort_unstable_by_key(|e| e.0);

    let mut out = Vec::with_capacity(5 + keep.len() * 4);
    out.push(2u8);
    out.extend_from_slice(&(keep.len() as u32).to_le_bytes());
    for (val, cnt) in keep {
        encode_u32(encode_i32_zigzag(val), &mut out);
        encode_u32(cnt, &mut out);
    }

    out.into_boxed_slice()
}

#[allow(unused_variables)]
pub fn query_idx(parameter: &Parameter, index: &[u8], query: &i32) -> Option<u64> {
    if index.len() < 5 {
        return None;
    }
    let version = index[0];
    let num_entries = {
        let mut b = [0u8; 4];
        b.copy_from_slice(&index[1..5]);
        u32::from_le_bytes(b) as usize
    };
    if version != 2 {
        return None;
    }

    let mut pairs: Vec<(i32, u32)> = Vec::with_capacity(num_entries);
    let mut pos = 5usize;
    for _ in 0..num_entries {
        let v_enc = decode_u32(index, &mut pos)?;
        let val = decode_i32_zigzag(v_enc);
        let cnt = decode_u32(index, &mut pos)?;
        pairs.push((val, cnt));
    }

    let mut lo = 0;
    let mut hi = pairs.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pairs[mid].0 == *query {
            return Some(pairs[mid].1 as u64);
        } else if pairs[mid].0 < *query {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    None
}
