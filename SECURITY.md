# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |
| Older snapshots | No |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security-sensitive reports.

Use GitHub private vulnerability reporting when it is enabled for the
[repository](https://github.com/YfengJ/steel-defect-detection). Include the
affected file or workflow, steps to reproduce, expected impact, and safe
proof-of-concept details. If private reporting is unavailable, open a minimal
public issue asking the maintainer to establish a private contact channel; do
not include vulnerability details in that issue.

## Scope

This project is a research and learning-oriented computer vision project. Security reports are most relevant for:

- unsafe file handling in scripts or the GUI
- dependency vulnerabilities that affect normal installation
- unsafe model or dataset loading behavior
- CI or repository configuration issues

Please do not include real production secrets, private datasets, or proprietary model weights in reports.

## Model Checkpoints

Only load model weights and PyTorch checkpoints from sources you trust. Legacy
checkpoints can contain serialized Python objects, and loading an untrusted
checkpoint may execute code in the current Python environment. Do not attach
private or untrusted weights to public issues.
