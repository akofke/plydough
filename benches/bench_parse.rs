use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use plydough::PlyData;
use std::fs::File;

pub fn bench_medium_ascii(c: &mut Criterion) {
    let bytes = std::fs::read_to_string("src/test_files/happy_vrip.ply").unwrap();

    let mut group = c.benchmark_group("Medium");
    group.throughput(Throughput::Bytes(bytes.as_bytes().len() as u64));
    group.sample_size(10);
    group.bench_function("parse", |b| {
        b.iter(|| {
            PlyData::parse_complete(bytes.as_bytes()).unwrap();
        })
    });
}

pub fn bench_binary(c: &mut Criterion) {
    let bytes = std::fs::read("src/test_files/buddha.ply").unwrap();

    let mut group = c.benchmark_group("Binary");
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.sample_size(10);
    group.bench_function("parse", |b| {
        b.iter(|| {
            PlyData::parse_complete(&bytes).unwrap();
        })
    });
}

criterion_group!(benches, bench_medium_ascii, bench_binary);
criterion_main!(benches);
