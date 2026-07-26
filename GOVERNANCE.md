# Governance

FrontierGuard DEA uses a maintainer-led governance model during its pre-alpha
stage. The goal is to make decisions transparently while preserving a coherent,
auditable mathematical core.

## Roles

### Lead maintainer

Vamsi Chinda ([@chindavamsi](https://github.com/chindavamsi)) is the founder,
lead maintainer, and initial release manager. The lead maintainer:

- sets project scope and release readiness;
- appoints and removes maintainers;
- has final responsibility for security and Code of Conduct enforcement;
- resolves decisions that do not reach consensus; and
- safeguards the project name, package indexes, and release credentials.

### Maintainers

Maintainers review contributions, triage issues, steward one or more technical
areas, and may merge changes within their responsibility. Maintainers must
demonstrate sustained, constructive contribution and sound judgment about
research validity, software quality, and community conduct.

A new maintainer is nominated publicly by an existing maintainer and appointed
by the lead maintainer after the community has at least seven days to comment.
Inactive maintainers may move to emeritus status after reasonable private
contact.

### Contributors

Anyone following the Code of Conduct may propose changes, review pull requests,
or participate in design discussions. Contributor status does not grant merge,
release, or administrative access.

## Decision process

Routine fixes and documentation changes are decided through pull-request
review. Material changes begin with a public issue and a written proposal when
they affect:

- a mathematical formulation or its default convention;
- the public API or serialized result schema;
- solver abstractions, validation gates, or numerical tolerances;
- supported Python versions or required dependencies;
- licensing, governance, security, or release infrastructure; or
- the definition of a supported method.

The preferred order is:

1. identify the user or research problem;
2. document alternatives, evidence, and compatibility implications;
3. seek rough consensus from affected contributors;
4. record the decision in the issue, method documentation, or an architecture
   decision record; and
5. implement through a reviewed pull request.

Consensus means no unresolved, evidence-backed objection remains after a
reasonable discussion period. It does not require unanimity. If a decision is
blocked, the lead maintainer documents the final choice and rationale.

## Scientific integrity

Feature count does not determine readiness. The validation gates in
`docs/VALIDATION.md` and status labels in `docs/FEATURE_MATRIX.md` are
governance controls: no maintainer may describe a method as supported until its
required evidence is present and reviewed.

Published conventions and compatibility differences must remain visible.
Maintainers may revert or disable a method when credible evidence puts its
correctness in doubt.

## Releases and access

Only designated release managers may publish GitHub or PyPI releases. Releases
must come from a reviewed tag, pass the repository's release workflow, and use
trusted publishing with a protected environment. No individual contributor
receives long-lived package-index credentials.

Administrative access follows least privilege. Changes to workflows,
`CODEOWNERS`, governance, or release metadata require review from the lead
maintainer while the project has only one maintainer.

## Amendments

Governance changes use the material-change process above. As the maintainer
group grows, the project intends to replace single-person final authority with
a documented multi-maintainer voting and succession process. The governance
model and active maintainers will be reviewed before version 1.0.
