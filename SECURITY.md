# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

Report security vulnerabilities to: security@aumos.ai

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if known)

We will acknowledge receipt within 48 hours and provide a resolution timeline within 7 days.

## Security Considerations

### Re-identification Risk
This service evaluates privacy risk in synthetic data. Re-identification attack simulations
are performed only on data within a single tenant's isolation boundary. Results are stored
with RLS enforcement and never cross tenant boundaries.

### Certificate Integrity
Compliance certificates are signed with a tenant-scoped key. Tampering with certificates
invalidates the signature. Store certificate URIs in MinIO with restricted access.

### Data Handling
- Source and synthetic datasets are referenced by URI, not stored inline
- Datasets in MinIO use tenant-prefixed bucket paths
- No PII is logged — only aggregate metrics and anonymized identifiers
