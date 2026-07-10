# Security Policy

## Reporting a Vulnerability

Please do not open a public GitHub issue for security-sensitive reports.

Use GitHub's private vulnerability reporting form:

- https://github.com/YfengJ/steel-defect-detection/security/advisories/new

If that form is unavailable, email `jfengy04@foxmail.com` with the subject
`[steel-defect-detection security]`. Include the affected file or workflow,
steps to reproduce, expected impact, and safe proof-of-concept details.

You should receive an acknowledgement within seven days. Fix timing depends on
severity, reproducibility, and whether an upstream dependency is involved.

## Scope

This project is a research and learning-oriented computer vision project. Security reports are most relevant for:

- unsafe file handling in scripts or the GUI
- dependency vulnerabilities that affect normal installation
- unsafe model or dataset loading behavior
- CI or repository configuration issues

Only load model weights from sources you trust. PyTorch checkpoint formats may
contain serialized Python objects, so a malicious `.pt` or `.pth` file can be a
security risk even when its filename looks normal.

Please do not include real production secrets, private datasets, or proprietary model weights in reports.
