# Releases

Pull requests and pushes to `main` run source checks, build the extension and validate its archive with Blender 4.2.7 and 5.2.2. Merging does not publish a release. Merge several focused changes before releasing if desired.

## Publish

From the `devdotkuru` account, open **Actions > Release > Run workflow**, select `main` and click **Run workflow**. No version or release-note input is needed.

The workflow captures `main` when dispatched. Later merges do not change that snapshot or cancel its publication. A snapshot already published is skipped; a snapshot older than published source is rejected.

The version uses the Europe/Paris calendar date and next unused daily counter. For example, `2026.10.2401` means the first allocation on October 24, 2026. Counters run from 01 to 99. Allocations must exceed the previous installed version, including the release preceding the public repository. Exhausted counters or a clock date below that version stop publication with an error. Routine contributions do not change versions manually.

The workflow updates `blender_manifest.toml` and `src/metadata.py` in a release-only child commit and reserves `v<version>`. Protected `main` stays unchanged. The tag reserves the version even if a later job fails. Each check job fetches that commit and rebuilds its package. Both Blender archive validations and source checks must pass before publication.

Release jobs use fresh runners and pinned inputs. Anders workflows do not use Actions caches, persistent dependency caches or uploaded Actions artifacts. Only source commit and version outputs pass between jobs; packages are rebuilt from source.

## Published assets

GitHub Releases contains:

- `anders_ugx_tools-<version>.zip`, the installable extension.
- Its `.zip.json` file with archive size, SHA-256 hash and version.
- `SHA256SUMS`, covering the ZIP, size/hash metadata and index.
- `index.json`, the Blender extension repository entry.

The workflow downloads uploaded assets and compares their complete bytes before publishing the draft. Existing assets are never overwritten. Retrying the same captured snapshot resumes its reserved tag and incomplete draft, verifies existing bytes and uploads only missing files. If the snapshot already published, rerunning skips it. A published version is preserved; corrections require a new pull request and release.

The first release includes installation guidance. Later notes list merged pull request titles and links since the preceding release, limited to the captured source snapshot.

The stable update URL is:

```text
https://github.com/newarcadia/anders/releases/latest/download/index.json
```

The addon ID remains `anders_ugx_tools`. Downloads and update metadata live exclusively in GitHub Releases. Routine releases require neither website deployment nor edits to download links.

## Maintainer permissions

Both maintainers may merge checked pull requests. The official Release workflow accepts only `devdotkuru`, including workflow retries. `kornman00` has GitHub Write access, which inherently also permits manual release management. This workflow restriction does not remove that GitHub role permission. See [GitHub repository roles](https://docs.github.com/en/organizations/managing-user-access-to-your-organizations-repositories/managing-repository-roles/repository-roles-for-an-organization).

For local development packages, use `python3 scripts/package.py --output out/anders.zip`. Local packaging does not allocate a version or publish a release. The source archives GitHub adds to releases are development snapshots, not installable extension ZIPs.
