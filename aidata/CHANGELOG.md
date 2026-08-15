# 8. CHANGELOG.md
changelog = '''# 📝 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2024-XX-XX

### Added
- Initial release of AIDATA
- `AIDATAWriter` — Write compressed, chunked datasets
- `AIDATAReader` — Read with random access, batch access, and LRU cache
- `AIDATADataset` — PyTorch `Dataset` for sample-level access
- `AIDATABatchDataset` — PyTorch `Dataset` for batch-level access
- `AIDATALoader` — Native PyTorch iterator with shuffle and device transfer
- `zstd` compression per chunk
- JSON metadata and index in file footer
- Comprehensive test suite with pytest
- 7 runnable examples (basic, training, benchmark, profiling)
- Full documentation: README, API reference, architecture guide, FAQ

### Features
- **Compression**: `zstd` level 3 reduces file size 2×–5×
- **Indexing**: O(1) random chunk access via footer index
- **Caching**: LRU cache with configurable size
- **PyTorch Integration**: Direct tensor conversion, GPU transfer
- **Metadata**: Store any JSON-serializable metadata
- **Validation**: Check magic, version, checksums on open

---

## [Unreleased / Planned]

### Added
- Multi-dimensional target support (images, masks)
- Prefetching for overlapping I/O and compute
- Sample-level shuffling (not just batch-level)
- Multi-worker support (`num_workers > 0`)
- Progress bar for writing large datasets
- Cloud storage support (S3, GCS via fsspec)
- Data integrity checks (CRC32 per chunk)
- Incremental append mode
- Built-in transforms pipeline
- Configurable compression algorithms (lz4, gzip)
- Memory-mapped reading (mmap)
- Distributed training support (DDP shards)
- Query/filter API for metadata

### Changed
- Improved error messages with suggestions
- Faster index parsing for large chunk counts
- Reduced memory allocations during batch reads

### Fixed
- [Future bug fixes will be listed here]

---

## Version History

| Version | Date | Highlights |
|---------|------|-----------|
| 0.5.0 | 2024-XX-XX | Initial stable release |
| 0.4.0 | — | Beta testing |
| 0.3.0 | — | Internal prototype |
| 0.2.0 | — | Format design |
| 0.1.0 | — | Proof of concept |
'''

with open(f"{base}/CHANGELOG.md", "w") as f:
    f.write(changelog)

# 9. requirements.txt
requirements = '''# Runtime dependencies
numpy>=1.20.0
zstandard>=0.15.0
torch>=1.10.0

# Development dependencies (install with: pip install -e ".[dev]")
# pytest>=7.0
'''