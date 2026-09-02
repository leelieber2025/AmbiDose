# Security Policy

## Supported Versions

AmbiDose is under active development. Security fixes are applied to the latest
released version on [PyPI](https://pypi.org/project/ambidose/); we do not maintain
long-term security support for older releases.

| Version | Supported |
| ------- | --------- |
| Latest release | :white_check_mark: |
| Older releases | :x: |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, use GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/leelieber2025/AmbiDose/security/advisories/new)
(repository → **Security** tab → **Report a vulnerability**).

Until the repository is public, email `leelieber@gmail.com`.

Include a description of the issue, steps to reproduce, and the affected
version.

## Scope

AmbiDose is a local single-cell analysis library (no network services, no
telemetry, no remote code execution by design). Relevant concerns include
unsafe deserialization of untrusted input files (HDF5 / h5ad) and
dependency vulnerabilities in packages it relies on.
