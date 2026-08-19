# Security Policy

## Supported versions

Security fixes are currently expected to target the latest release on the default branch.

Because AIDATA is pre-1.0, users should pin versions in production and test dataset compatibility before upgrades.

## Scope

Relevant security reports include issues such as:

- parser crashes caused by malformed AIDATA files
- uncontrolled memory allocation caused by malformed metadata/index values
- path or filesystem handling vulnerabilities
- decompression-related denial of service
- unsafe deserialization
- security-sensitive dependency vulnerabilities

A normal data-corruption bug is also worth reporting, but label it as corruption/reliability rather than security when appropriate.

## Out of scope

AIDATA is not an encryption or authentication system. CRC32 and metadata SHA-256 are integrity mechanisms for corruption detection, not protection against malicious modification.

## Reporting

Do not publish an unpatched security vulnerability in a public issue.

When this repository has a private security-reporting channel enabled, use it. Otherwise, contact the project maintainers privately through the repository owner's verified contact method and provide:

- affected version
- Python version
- operating system
- minimal reproduction
- traceback or crash details
- whether an attacker must control the AIDATA file
- impact assessment

Please do not include secrets or personal data in reports.
