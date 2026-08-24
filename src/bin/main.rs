use skip_list::parameter::Parameter;
use skip_list::user::{build_idx, query_idx};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: <program> <file> <factor-size> <factor-skip>");
        std::process::exit(1);
    }

    let parameter = Parameter {
        factor_size: args[2].parse().unwrap_or(1),
        factor_skip: args[3].parse().unwrap_or(1),
    };
    let file_stem = &args[1];

    let data_path = PathBuf::from(format!("./data/{}.data", file_stem));
    let query_path = PathBuf::from(format!("./data/{}.query", file_stem));

    let data = load_data(&data_path)?;
    let mut indexes: Vec<Box<[u8]>> = Vec::with_capacity(data.len());
    let mut total_index_size_bytes = 0usize;

    let statistics = analyze_data(&data);

    print_statistics(&statistics);

    for block in data.iter() {
        let idx = build_idx(&parameter, block);
        total_index_size_bytes += idx.len();
        indexes.push(idx);
    }

    let queries = load_query(&query_path)?;
    let mut n_skipped = 0u64;

    for query in queries.as_slice() {
        let mut expected_result = 0u64;
        let mut user_result = 0u64;

        for block_i in 0..data.len() {
            let mut block_result = 0u64;
            for num in data[block_i].as_slice() {
                if query == num {
                    block_result += 1;
                }
            }
            expected_result += block_result;

            match query_idx(&parameter, &indexes[block_i], query) {
                None => {
                    user_result += block_result;
                }
                Some(idx_hint) => {
                    user_result += idx_hint;
                    n_skipped += 1;
                }
            }
        }
        assert_eq!(expected_result, user_result, "Wrong answer!");
    }
    let total_index_size_kb = total_index_size_bytes / 1024;
    let max_score = parameter.factor_skip * queries.len() as i64 * data.len() as i64;
    let score = (n_skipped as i64 * parameter.factor_skip)
        - (total_index_size_kb as i64 * parameter.factor_size);
    println!(
        "storage size: {} {}",
        total_index_size_kb,
        total_index_size_kb as i64 * parameter.factor_size
    );
    println!(
        "num skips: {} {}",
        n_skipped,
        n_skipped as i64 * parameter.factor_skip
    );
    println!("total score: {}", score / max_score);
    Ok(())
}

fn load_data(file_path: &Path) -> io::Result<Vec<Vec<i32>>> {
    let mut file = BufReader::new(File::open(file_path)?);

    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)?;

    let n_block = u64::from_le_bytes(buf8) as usize;
    file.read_exact(&mut buf8)?;

    let chunk_size = u64::from_le_bytes(buf8) as usize;
    let mut blocks: Vec<Vec<i32>> = Vec::with_capacity(n_block);

    let mut buf4 = [0u8; 4];
    for _block_idx in 0..n_block {
        let mut chunk = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            file.read_exact(&mut buf4)?;
            chunk.push(i32::from_le_bytes(buf4));
        }
        blocks.push(chunk);
    }

    Ok(blocks)
}

fn load_query(file_path: &Path) -> io::Result<Vec<i32>> {
    let mut file = BufReader::new(File::open(file_path)?);

    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8)?;
    let n = u64::from_le_bytes(buf8) as usize;

    let mut data = Vec::with_capacity(n);
    let mut buf4 = [0u8; 4];
    for _ in 0..n {
        file.read_exact(&mut buf4)?;
        data.push(i32::from_le_bytes(buf4));
    }

    Ok(data)
}
use std::collections::HashMap;

#[derive(Debug)]
pub struct Statistics {
    pub total_values: usize,
    pub distinct_count: usize,
    pub min: i32,
    pub max: i32,
    pub mean: f64,
    pub median: f64,
    pub mode: Vec<i32>,
    pub mode_frequency: usize,
    pub frequency_map: HashMap<i32, usize>,
    pub frequency_buckets: Vec<FrequencyBucket>,
    pub percentiles: Percentiles,
}

#[derive(Debug)]
pub struct FrequencyBucket {
    pub frequency: usize,
    pub count: usize,
}

#[derive(Debug)]
pub struct Percentiles {
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

pub fn analyze_data(blocks: &[Vec<i32>]) -> Statistics {
    let mut all_values: Vec<i32> = blocks.iter().flatten().copied().collect();

    if all_values.is_empty() {
        panic!("No data to analyze");
    }

    let total_values = all_values.len();

    let mut frequency_map: HashMap<i32, usize> = HashMap::new();
    for &val in &all_values {
        *frequency_map.entry(val).or_insert(0) += 1;
    }

    let distinct_count = frequency_map.len();

    let min = *all_values.iter().min().unwrap();
    let max = *all_values.iter().max().unwrap();

    let sum: i64 = all_values.iter().map(|&x| x as i64).sum();
    let mean = sum as f64 / total_values as f64;

    all_values.sort_unstable();
    let median = percentile(&all_values, 50.0);
    let percentiles = Percentiles {
        p25: percentile(&all_values, 25.0),
        p50: median,
        p75: percentile(&all_values, 75.0),
        p90: percentile(&all_values, 90.0),
        p95: percentile(&all_values, 95.0),
        p99: percentile(&all_values, 99.0),
    };

    let max_frequency = *frequency_map.values().max().unwrap();
    let mode: Vec<i32> = frequency_map
        .iter()
        .filter(|(_, &freq)| freq == max_frequency)
        .map(|(&val, _)| val)
        .collect();

    let mut freq_distribution: HashMap<usize, usize> = HashMap::new();
    for &freq in frequency_map.values() {
        *freq_distribution.entry(freq).or_insert(0) += 1;
    }

    let mut frequency_buckets: Vec<FrequencyBucket> = freq_distribution
        .into_iter()
        .map(|(frequency, count)| FrequencyBucket { frequency, count })
        .collect();
    frequency_buckets.sort_by_key(|b| b.frequency);

    Statistics {
        total_values,
        distinct_count,
        min,
        max,
        mean,
        median,
        mode,
        mode_frequency: max_frequency,
        frequency_map,
        frequency_buckets,
        percentiles,
    }
}

fn percentile(sorted_data: &[i32], p: f64) -> f64 {
    let n = sorted_data.len();
    let rank = (p / 100.0) * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;

    if lower == upper {
        sorted_data[lower] as f64
    } else {
        sorted_data[lower] as f64 * (1.0 - fraction) + sorted_data[upper] as f64 * fraction
    }
}

pub fn print_statistics(stats: &Statistics) {
    println!("\n=== Statistical Analysis ===");
    println!("Total values: {}", stats.total_values);
    println!("Distinct values: {}", stats.distinct_count);
    println!("Range: {} to {}", stats.min, stats.max);
    println!("Mean: {:.2}", stats.mean);
    println!("Median: {:.2}", stats.median);
    println!(
        "Mode: {:?} (appears {} times)",
        stats.mode, stats.mode_frequency
    );

    println!("\n--- Percentiles ---");
    println!("25th: {:.2}", stats.percentiles.p25);
    println!("50th: {:.2}", stats.percentiles.p50);
    println!("75th: {:.2}", stats.percentiles.p75);
    println!("90th: {:.2}", stats.percentiles.p90);
    println!("95th: {:.2}", stats.percentiles.p95);
    println!("99th: {:.2}", stats.percentiles.p99);

    println!("\n--- Frequency Distribution ---");
    println!("(How many distinct values appear X times)");
    for bucket in &stats.frequency_buckets {
        println!(
            "  Frequency {}: {} distinct value(s)",
            bucket.frequency, bucket.count
        );
    }

    println!("\n--- Top 10 Most Frequent Values ---");
    let mut freq_vec: Vec<_> = stats.frequency_map.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));
    for (i, (&value, &count)) in freq_vec.iter().take(10).enumerate() {
        println!(
            "  {}. Value {} appears {} times ({:.2}%)",
            i + 1,
            value,
            count,
            (count as f64 / stats.total_values as f64) * 100.0
        );
    }
}
