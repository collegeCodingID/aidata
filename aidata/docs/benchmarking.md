# Benchmarking AIDATA

Benchmark claims are only useful when the workload and methodology are reproducible.

## Run repository benchmarks

```bash
python examples/benchmark.py
python examples/training_benchmark.py
```

## Compare against alternatives

For a useful storage benchmark, compare at least:

- NumPy `.npy`
- NumPy `.npz`
- CSV where applicable
- Parquet where applicable
- HDF5 where applicable
- AIDATA

Do not compare formats with incompatible representations or unfair preprocessing.

## Measure

### Storage

- resulting file size
- compression ratio

### Write path

- total write time
- samples/second
- MB/second

### Read path

- sequential full-dataset read
- random sample read
- contiguous batch read
- repeated access with cache

### Training

- batches/second
- samples/second
- GPU utilization where applicable
- CPU utilization
- peak RAM
- time spent waiting for data

## Reproducibility checklist

Record:

```text
CPU
RAM
Storage device
Operating system
Python version
NumPy version
PyTorch version
AIDATA version
dataset shape
dtypes
chunk size
compression mode
compression level
batch size
num_workers
```

## Avoid misleading benchmarks

Do not publish only the fastest result. Report the complete workload and include the competing format's configuration.

AIDATA is optimized for specific sample-oriented access patterns; no single storage format is fastest for every workload.
