# Security policy

FrontierGuard DEA values both software security and analytical integrity.

## Supported versions

The project is pre-alpha and has not made a stable release. Security fixes are
applied to the latest code on `main` and will be included in the next release.
Older pre-alpha snapshots are not supported.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Unreleased local snapshots | No |

This policy will change to a release-based support table before version 1.0.

## Report a vulnerability privately

Do not open a public issue, discussion, or pull request for a suspected
vulnerability.

Use GitHub's
[private vulnerability reporting form](https://github.com/chindavamsi/frontierguard-dea/security/advisories/new).
Include:

- affected versions or commit;
- impact and realistic threat scenario;
- minimal reproduction or proof of concept;
- any known workarounds;
- whether the issue has been disclosed elsewhere; and
- how you would like to be credited.

Remove credentials, confidential datasets, personal information, and
unnecessary exploit data before submitting the report.

## What to expect

The maintainer aims to:

1. acknowledge a complete report within 7 calendar days;
2. provide an initial assessment within 14 calendar days;
3. keep the reporter informed when the status materially changes; and
4. coordinate disclosure after a fix or mitigation is available.

These are targets for a volunteer-maintained project, not a service-level
agreement. Please allow a reasonable remediation window before public
disclosure.

## Scope

Security reports include dependency or packaging compromise, arbitrary code
execution, unsafe deserialization, path traversal, sensitive-data exposure,
workflow credential exposure, and a reproducible integrity failure that an
attacker could exploit.

Ordinary numerical disagreement, model-definition questions, and solver
infeasibility without a security impact should use the bug-report template.
Incorrect analytical results are still treated seriously; the private channel
is required only when public disclosure would create a concrete security or
privacy risk.

## Safe-harbor intent

Good-faith research that avoids privacy violations, data destruction, service
degradation, social engineering, and unauthorized access will not be met with
legal action by the project maintainer. Stop testing and report immediately if
you encounter sensitive data. This statement cannot authorize testing of
third-party infrastructure or override applicable law.
