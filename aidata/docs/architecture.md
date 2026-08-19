# Architecture

## Components

```text
                 ┌─────────────────┐
                 │ AIDATAWriter    │
                 └────────┬────────┘
                          │
                          ▼
                    AIDATA v1 file
                          │
                          ▼
                 ┌─────────────────┐
                 │ AIDATAReader    │
                 └───────┬─────────┘
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          indexing     caching    validation
             │           │           │
             └───────────┼───────────┘
                         ▼
                  NumPy arrays
                         │
                         ▼
                PyTorch integration
```

## Writer

The writer:

1. validates input arrays and metadata
2. normalizes supported dtype endianness
3. creates a sibling temporary file
4. writes the header and metadata
5. writes X/y chunks
6. calculates CRC32 values
7. writes the JSON index and footer
8. patches the chunk count in the header
9. flushes and fsyncs the temporary file
10. atomically replaces the destination

This prevents a partially written destination from being mistaken for a successfully completed dataset.

## Reader

The reader:

1. opens the file read-only
2. validates the header
3. parses and validates metadata
4. validates metadata SHA-256
5. locates the footer
6. validates the index boundaries
7. parses and validates all chunk descriptors
8. serves samples/batches through indexed reads

## Cache

`AIDATAReader` uses an LRU cache for decompressed chunks. The cache is intentionally bounded by the configured number of cached chunks.

## PyTorch layer

The PyTorch layer sits above the core reader. It is optional so users who only need NumPy storage do not need to install PyTorch.

The integration avoids serializing open file handles into worker processes.

## Design boundaries

The core should remain responsible for:

- binary format
- I/O
- validation
- indexing
- compression
- integrity
- basic caching

Higher-level functionality should be separate where practical:

- profiling
- quality scoring
- visualization
- drift detection
- dataset versioning
- experiment tracking

This keeps the package understandable and reduces dependency pressure.
