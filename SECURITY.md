# Security Policy

## Supported versions

`quantrules` is pre-1.0. Security fixes are applied to the latest released version only.

| Version | Supported |
| --- | --- |
| Latest `0.x` release | Yes |
| Anything older | No |

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report privately through
[GitHub Security Advisories](https://github.com/Milton-Analytics-LLC/quantrules/security/advisories/new),
or by email to **security@withmilton.ai**.

Please include:

- what the issue is and why you believe it is a security problem,
- the version or commit affected,
- the smallest reproduction you can manage.

We aim to acknowledge a report within three business days and to give you an assessment
and a remediation plan within ten. We will credit you in the advisory unless you would
rather we did not.

## Scope

`quantrules` is a numerical library. It performs no network access, opens no sockets,
reads no files, executes no user-supplied code, and handles no credentials. The realistic
security surface is therefore small, and is mostly:

- **Supply chain** - a compromised release artefact, or a malicious dependency. Releases
  are built in GitHub Actions and published to PyPI with trusted publishing, so no
  long-lived API token exists to steal.
- **Denial of service** - an input that causes unbounded memory or time consumption.
- **Deserialisation** - `from_mapping` constructs configuration objects from plain
  mappings. It never evaluates code, but do not feed it untrusted input without
  validating the source.

!!! note

    A *numerical* error - a wrong formula, an incorrect result - is a bug, not a
    vulnerability. Please report those as normal issues so they can be discussed in
    public.

## What is not a vulnerability

Losing money is not a security issue. `quantrules` is not investment advice and makes no
claim that its output is profitable. See [DISCLAIMER.md](DISCLAIMER.md).
