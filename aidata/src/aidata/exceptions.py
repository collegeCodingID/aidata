class AIDATAError(Exception):
    """Base exception for AIDATA."""


class InvalidAIDATAFile(AIDATAError):
    """Raised when an AIDATA file is invalid or corrupted."""


class UnsupportedVersion(AIDATAError):
    """Raised when an unsupported AIDATA version is detected."""
