# Security Policy

This repository is research software. It is not intended for safety-critical,
medical, financial, or production security-sensitive use.

## Supported versions

Only the current `main` branch is maintained.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| older commits/releases | No |

## Reporting a vulnerability

Please report security issues privately through GitHub Security Advisories if
available for this repository. If that is not available, contact the repository
owner through GitHub.

Do not open a public issue for a vulnerability until it has been reviewed.

## Scope

Security reports that are useful for this project include:

- unsafe deserialization or file handling;
- dependency issues that affect normal installation or test workflows;
- code execution risks introduced by examples, benchmarks, or docs.

Out of scope:

- numerical inaccuracy without a security impact;
- benchmark disagreement;
- theoretical objections to the research framing.
