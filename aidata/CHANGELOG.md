# Changelog

All notable changes to AIDATA are documented here.

## [0.5.6] - 2026-08-19

### Added

- Deterministic corruption mutation/fuzz-style tests.
- Random truncation tests.
- Public API compatibility tests.
- GitHub Actions testing for Python 3.10, 3.11, 3.12, 3.13, and 3.14.
- Expanded project documentation and contribution/security files.

### Changed

- Standardized public error messages for common argument validation failures.
- Preserved detailed decompression failure reasons in `InvalidAIDATAFile` messages.
- Cleaned compatibility, error-handling, and hardening documentation.

## [0.5.5]

### Fixed

- Fixed `examples/train_demo.py` to use the canonical batch dataset API.
- Fixed strict zlib validation to reject trailing bytes after a compressed payload.

### Changed

- Consolidated PyTorch dataset implementations around `aidata.dataset`; the legacy integrations import path provides compatibility aliases.
- Expanded corruption/hardening tests for trailing compressed data and truncated files.
- Updated documentation and version metadata.

## [0.5.4]

### Changed

- Replaced the mandatory `zstandard` dependency with Python standard-library `zlib` compression.
- Added bounded zlib decompression and strict compression-level validation.
- Added `compression_level` to reserved metadata keys.
- Fixed PyTorch batch dataset integration and examples.
- Added regression coverage for compression and PyTorch integration.

## [0.5.3]

### Fixed

- Made `AIDATADataset` safe for PyTorch `DataLoader(num_workers > 0)`.
- Added worker-local reader/cache lifecycle for fork and spawn.
- Prevented open file handles from being serialized into workers.
- Made PyTorch optional rather than a core runtime dependency.
- Added strict validation for compression configuration.
- Added validation for zero-sized sample dimensions.
- Added metadata and index size limits required by v1 binary structures.
- Added strict validation that the index ends immediately before the footer.
- Added checksum range validation.
- Added persistent reader cleanup.

### Changed

- Documented that `AIDATALoader(shuffle=True)` shuffles batch order rather than individual samples.
- Improved worker-safe PyTorch integration.

## [0.5.2]

- Hardened file writing with atomic replacement.
- Added CRC32 payload checksums.
- Added metadata SHA-256 integrity checking.
- Added strict metadata/index/file-bound validation.
- Added persistent reader cleanup.
- Added multidimensional target support.

## [0.5.1]

- Initial AIDATA release.
