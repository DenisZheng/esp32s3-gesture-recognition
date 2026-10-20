# GitHub onboarding — completed on 2026-10-20

- Repository: https://github.com/DenisZheng/esp32s3-gesture-recognition
- Visibility: **Public**, explicitly approved by repository owner.
- Default branch: `main`; initial source/document history imported through GitHub connector.
- Actions: `.github/workflows/p0-ci.yml`, mandatory status context **`p0-contract`**, verified green on `main`.
- Active Ruleset: [Protect main](https://github.com/DenisZheng/esp32s3-gesture-recognition/rules/25458714). Direct API inspection confirmed target default branch, no bypass actors, deletion and non-fast-forward blocked, PR required, review-thread resolution, linear history, mandatory `p0-contract` with strict up-to-date checking, zero required human approvals.
- GitHub connector has admin/push access to this existing repository; **no ruleset administrative write action is available**, but owner completed setup via GitHub UI.

## Future changes

1. Create topic branch from the latest `main` (e.g. `feat/p1-...`).
2. Commit source/document/test changes to the topic branch, not `main`.
3. Open PR, ensure `p0-contract` (and any future specific jobs) completes successfully.
4. Merge with a linear-history-compatible method (Squash or Rebase); use real experiment evidence for human-only P1–P4 gates.
5. Do not treat passing source CI as training, license, hardware or data acceptance.
