# Contributing

Anders is developed by New Arcadia and its contributors under GPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Build locally

Use Python 3.11 or newer and Blender 4.2 or newer.

```sh
git clone https://github.com/newarcadia/anders.git
cd anders
python3 -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/check.py
python scripts/package.py --output out/anders.zip
```

Install the ZIP through Blender Preferences > Get Extensions > Install from Disk. See [development](docs/development.md) for source installation and reloads. A native toolkit build is not required.

## Report an issue

Include the Anders version, Blender version, operating system, game, reproduction steps and exact error text. Screenshots help with preview and interface issues. Do not upload retail assets, game dumps, credentials or private builds.

## Submit a change

Any focused pull request is welcome. An issue is not required first; discuss large or incompatible changes in an issue before investing substantial work.

1. Outside contributors fork the repository. Maintainers use short-lived branches such as `feature/material-preview`, `fix/uax-export` or `docs/installation`.
2. Keep changes focused and preserve native asset data and compatibility. Do not edit version numbers for routine changes.
3. Open a pull request against `main`. Use the template to describe the problem, resulting behavior, checks actually performed and anything unverified.
4. Keep the branch current with `main` and resolve review conversations. Required source checks, package builds and Blender 4.2.7 and 5.2.2 archive validations must pass.

Both `devdotkuru` and `kornman00` may squash-merge pull requests. A second approving review is not required. Direct pushes to `main`, force pushes and deletion of `main` are prohibited, including for administrators. Completed branches are deleted automatically.

Archive validation does not establish correct editing, rendering or in-game behavior. Describe the affected Blender workflows you checked and their versions separately from in-game results.

Keep copyright, license and third-party notices intact. Contributions must be work you have permission to distribute under GPL-3.0-or-later. Game assets and trademarks are not covered by that permission.

See the [repository layout](docs/repository-layout.md), [release workflow](docs/releases.md) and [official manual](https://docs.new-arcadia.net/anders).
