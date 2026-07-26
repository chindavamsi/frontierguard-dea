# Release process

This runbook is for FrontierGuard DEA release managers. A release is complete
only when the same distributions are validated, attested, published to PyPI,
and attached to the GitHub Release.

## One-time repository setup

1. Create a protected GitHub environment named `pypi`.
2. Add Vamsi Chinda (`@chindavamsi`) as a required reviewer for that
   environment.
3. On PyPI, create the `frontierguard-dea` project or use a pending trusted
   publisher with:
   - owner: `chindavamsi`
   - repository: `frontierguard-dea`
   - workflow: `release.yml`
   - environment: `pypi`
4. Protect `main`, require the CI jobs, require review, dismiss stale approvals,
   and block force pushes.
5. Restrict workflow changes through `CODEOWNERS` review.

The workflow uses OpenID Connect trusted publishing. Do not create or store a
PyPI API token in GitHub.

## Prepare a release

1. Confirm every method marked supported in `docs/FEATURE_MATRIX.md` satisfies
   `docs/VALIDATION.md`.
2. Update the version in both `pyproject.toml` and
   `src/frontierguard/_version.py`.
3. Move user-facing entries from `Unreleased` to a versioned section in
   `CHANGELOG.md`, including the release date.
4. Update `CITATION.cff` with the same version and release date.
5. Run:

   ```bash
   uv lock --check
   uv run ruff format --check .
   uv run ruff check .
   uv run mypy
   uv run pytest --cov=frontierguard --cov-report=term-missing
   uv build --no-sources --clear
   uvx --from twine==6.2.0 twine check --strict dist/*
   ```

6. Open and merge a release-preparation pull request. Do not publish from an
   unreviewed working tree.

## Publish

1. Create an annotated tag whose name exactly matches `v` plus the package
   version, for example `v0.1.0a0`.
2. Push the tag and create a GitHub Release from it.
3. Mark unstable releases as pre-releases, write release notes from the
   changelog, and publish the GitHub Release.
4. Review and approve the `pypi` environment deployment.
5. Wait for the `Publish release` workflow to finish.

The workflow independently repeats formatting, lint, typing, tests, coverage,
version/tag matching, package build, and metadata validation. It then generates
GitHub build-provenance attestations, publishes with PyPI trusted publishing,
and attaches the exact distributions to the GitHub Release.

## Verify

1. Compare SHA-256 hashes shown by PyPI, the workflow, and the GitHub Release.
2. Install from PyPI in a clean environment:

   ```bash
   uv run --isolated --no-project --with frontierguard-dea==VERSION python -c \
     "import frontierguard; print(frontierguard.__version__)"
   ```

3. Confirm the PyPI project metadata, README, license, Python classifiers, and
   repository links render correctly.
4. Open a new `Unreleased` section in the changelog if needed.

PyPI distributions are immutable. If a release is wrong, do not overwrite its
tag or artifacts. Yank the affected version when appropriate, document the
reason, fix forward, and publish a new version.
