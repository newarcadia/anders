# Repository layout

| Path | Contents |
| --- | --- |
| `__init__.py` | Blender extension registration and development reloads |
| `anders.py` | Compatibility entry point for existing scripts |
| `blender_manifest.toml` | Extension metadata and packaged paths |
| `src/` | Implementation grouped by asset format and authoring feature |
| `scripts/` | Source checks, packaging, GitHub releases and development installation |
| `.github/` | Repository checks, release workflow and contribution templates |
| `docs/` | Developer, repository and release guidance |

Both `__init__.py` files are needed: the root file registers the extension, while `src/__init__.py` defines its implementation package.

The toolkit owns its Visual Studio project, application icons and Blender terrain bridge outside this repository. They are not part of the Anders extension. The extension archive contains only the paths listed in its manifest.

Keep generated output, game dumps and local investigations out of the repository. See [development](development.md) and [releases](releases.md).

The [official manual](https://docs.new-arcadia.net/anders) is maintained separately. Private publication checklists and retired toolkit resources are not part of this repository.
