# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue.
2. Email the maintainer directly with details.
3. Allow reasonable time for a fix before public disclosure.

lfm-physics is a scientific simulation library with no network services,
authentication, or user data handling.  Security concerns are limited to
arbitrary code execution via pickle/eval (we use only numpy `.npz` for
checkpoints, which is safe).
