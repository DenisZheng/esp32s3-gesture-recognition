# One-time GitHub setup (minimal user actions)

P0 local assets are ready, but the connected GitHub App exposes no repository-creation action and the current workspace has no `gh` binary. Therefore there is **no remote repo and no remote commit**, and the following is the shortest legitimate manual unblock.

1. Open **https://github.com/new**. Owner: `DenisZheng`. Repository name: `esp32s3-gesture-det`. Description: `Self-trained 10-class gesture detection on ESP32-S3 using ESPDet-Pico and HaGRID.` Choose **Private**. **Do not** initialize README, `.gitignore` or license; no Fork. Click **Create repository**.
2. Ensure the ChatGPT GitHub connection has access to the new Private repo (if the App is configured for selected repositories, add this repository to the installation). Then in chat request importing the P0 scaffold through the GitHub connector. The GitHub connector supports writing files, commits, branches and PRs **to existing repositories**. If instead pushing the ZIP locally, extract the ZIP with its `.git` directory, then in the extracted root execute:

```bash
git remote add origin https://github.com/DenisZheng/esp32s3-gesture-det.git
git push -u origin main
```

3. After Actions produces a green P0 check, open repository **Settings -> Rules -> Rulesets -> New branch ruleset** (or branch protection if the plan/UI differs). Target `main`; require pull request before merge, prevent force pushes and branch deletions; require the exact check `p0-contract` only after it appears on GitHub and set it mandatory; optionally require conversation resolution. Keep pull-request bypass rules minimal. The GitHub connector lacks an administrative write action for rulesets, so this needs GitHub UI/admin rights.
4. Verify `Settings -> General` reports `Private`, branch list shows `main`, **Actions** shows `P0 offline guards` green, and direct push to `main` is blocked. Record verified GitHub commit SHA and CI run URL; then P0 can transition from BLOCKED once its license risk decision and class policy are documented.

**Do not give a fabricated GitHub URL as a live repo**. `https://github.com/DenisZheng/esp32s3-gesture-det` is the **intended** URL until GitHub `get_repo` successfully resolves it.
