# AIDATA v1 format specification

This document describes the current AIDATA v1 container implemented by version 0.5.6.

## High-level layout

```text
HEADER
METADATA
CHUNK_0
CHUNK_1
...
CHUNK_N
INDEX
FOOTER
```

## Header

The header is a fixed-width binary structure defined by `src/aidata/format.py`.

It records:

- AIDATA magic value
- format version
- metadata byte length
- chunk count

The reader validates all fixed-width reads before unpacking them.

## Metadata

Metadata is UTF-8 JSON. It describes the arrays and storage configuration.

Important fields include:

```text
format
version
samples
features
x_shape
y_shape
x_dtype
y_dtype
x_ndim
y_ndim
compression
compression_level
chunk_size
checksum
metadata_sha256
```

User metadata is allowed, but reserved AIDATA keys cannot be overwritten.

The metadata hash is calculated over canonical JSON with the `metadata_sha256` field excluded when validating an existing file.

## Chunks

Each chunk contains:

```text
X compressed/raw bytes
Y compressed/raw bytes
```

The index records:

```text
start
end
x_offset
y_offset
x_size
y_size
x_raw_size
y_raw_size
x_crc32
y_crc32
```

The X and y byte ranges are contiguous and are checked against the index boundary.

## Compression

AIDATA 0.5.6 supports:

- `none`
- `zlib`

When zlib is enabled, compression levels are `1..9`.

The reader checks that the decompressed output has the expected raw size and that the zlib stream is complete without trailing bytes.

## Index

The index is UTF-8 JSON containing a `chunks` list.

The footer identifies the index offset and size. The reader requires the index to end immediately before the footer.

A JSON index is intentionally simple and debuggable. It is not optimal for extremely large chunk counts; a future format revision may use a compact fixed-width index.

## Footer

The footer stores:

- index offset
- index size

The reader uses this to locate the index from the end of the file.

## Integrity model

Integrity is checked at two levels:

1. metadata SHA-256 protects metadata consistency
2. per-payload CRC32 detects accidental/corrupt payload changes

These checks are designed to detect corruption. They do not provide authenticated security against a malicious party who can rewrite the file and recompute the checksums.

## Compatibility

The v1 format is pre-1.0 and should be considered subject to documented evolution. Future format changes should use an explicit version and compatibility tests.
