class AIDATAError(Exception):
    """Base exception for AIDATA errors."""


class InvalidAIDATAFile(AIDATAError):
    """Raised when an AIDATA file is malformed or corrupted."""


class UnsupportedVersion(AIDATAError):
    """Raised when an AIDATA file uses an unsupported format version."""
