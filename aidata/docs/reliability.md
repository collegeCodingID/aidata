# Reliability and security

## What AIDATA protects against

The reader is designed to reject common accidental corruption cases:

- truncated files
- invalid magic/version
- invalid metadata
- metadata hash mismatch
- malformed index
- invalid chunk ranges
- out-of-bounds offsets and sizes
- checksum mismatch
- incomplete zlib streams
- trailing bytes after compressed members
- inconsistent raw sizes

## Atomic writes

The writer creates a sibling temporary file and uses `os.replace` after a successful write. It also flushes and fsyncs the temporary file before replacement.

This is intended to reduce the chance of exposing an incomplete destination after an interrupted write.

## CRC32 is not authentication

CRC32 detects many accidental changes but is not a cryptographic signature.

Do not interpret this as protection against a malicious file author. A malicious party can modify payload data and calculate a new CRC32.

## SHA-256 metadata integrity

Metadata includes a SHA-256 digest calculated over canonical metadata. This catches accidental or inconsistent metadata changes.

It is still not an authenticated signature because the same party that modifies metadata can recompute the digest.

## Untrusted files

AIDATA files should be treated as untrusted input. The reader performs extensive structural validation, but no binary parser should be considered a sandbox.

If your application processes files from unknown parties, isolate parsing appropriately and keep the library updated.

## Fuzzing

The repository includes deterministic mutation/truncation tests that exercise headers, metadata, payloads, index data, footer data, and file length changes.

For deeper fuzzing, use an external fuzzing framework against the public reader boundary.

## Security reporting

See the repository's `SECURITY.md` for the reporting process.
