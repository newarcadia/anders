# Anders development

Anders is a Python package. Install its extension ZIP, or keep the complete source directory together when loading it from a development launcher. Keep the complete package together.

## Source layout

| Path | Responsibility |
| --- | --- |
| `__init__.py` | Extension entry point and development reload |
| `anders.py` | Compatibility entry point for existing package scripts |
| `src/formats/` | Binary readers, ECF containers, BDT material trees, Granny data and pointer-graph validation, UFX headers and XMB |
| `src/ugx/` | Model records, vertex encoding, imports, authoring, source patches, damage bindings and validation |
| `src/animation/` | UAX files, editable actions, sampling, playback and animation operators |
| `src/textures/` | DDS/DDX codecs, compression, image loading, path lookup and export |
| `src/materials/` | HW1/HW2 material data, shader groups, node bindings, baking and migration |
| `src/porting/` | Removing source metadata and converting HW1 assets to HW2 |
| `src/visuals/` | VIS associations, bone markers, animation links, damage templates and paired exports |
| `src/ui/` | Preferences, panels and node controls |
| `src/registration.py` | Registration order, menus, handlers, timers and session cache cleanup |
| `scripts/` | Source checks, packaging and development installation |

Binary formats use bytes and plain records. They do not import `bpy`, `bmesh` or `mathutils`. Blender-specific transforms and scene mutation belong in the authoring modules. Operators invoke those operations and report their results; codecs do not obtain data from the active scene.

Imported records retain original bytes, offsets and unknown fields. Blender objects, materials and actions own user edits and their saved source metadata. Export constructs a temporary model from those inputs. Export proxies have an explicit removal path. Texture lookup and HW1 source-material caches belong to their feature modules and are cleared on file load and addon teardown.

`ugx/snapshots.py` records source-linked mesh buffer hashes at import. Export compares topology, UVs, native attributes, materials, weights, section ownership and model transforms before bypassing the combined mesh proxy. Position edits patch native section buffers and incident shading directly. Other edits use the full exporter. Exported buffers are reparsed and validated before files are replaced. Reusing decoded vertex values requires exact native buffer and layout equality and produces independent vertex records. Missing snapshots and externally rebased sources use the full path. Active point or corner colour attributes participate in the snapshot. `ugx/color_scene.py` prepares native colour-layout edits before skin promotion, and `ugx/color_layout.py` retains the other vertex channels. `materials/color_inputs.py` records verified native HW2 colour input signatures and header hashes. The reader uses the verified primary skin shader contract when byte candidates are ambiguous. Layout preparation serializes the matching shader fields before reparsing the temporary source, and refreshes source offsets without discarding authored material arrays.

VIS state links retain their model and exit mode. Shared looping and held uses have separate Actions with native boundary sampling. Export preserves shared source bytes until their edits differ, then writes a separate resource and changes only the matching links. Nearby UAX discovery keeps the original VIS states. Animation root aliases belong to each Action so native-root clips can coexist with compiled attachment-root clips.

Automatically linked UAX actions retain native source snapshots until selected. The loading queue prepares active and default clips, and the native Action Editor and NLA trigger preparation on use. Sampling prepares sorted key arrays once and finds intervals by binary search. Versioned curve fingerprints read numeric key data in bulk, while older saved signatures keep their original comparison algorithm. Texture previews remain editable in memory during import and pack on scene save. The animation-loading and export-buffer regressions cover selection, metadata edits, scalar decoder parity, consecutive exports and saved scenes.

`animation/channels.py` prepares native spline controls independently of scene state. `animation/local_sampling.py` compiles rigid local basis changes; other rigs use the full pose conversion. `animation/curve_fitting.py` fits Bezier spans against sampled transforms, retains one-sided limits and encodes native cubic controls. `animation/keyframes.py` converts those spans to editable keys and handles, including step boundaries. Native knots and interval probes constrain smooth fits; only genuine discontinuities force their own boundaries. Export samples evaluated fractional frames, including edits to handles, while unchanged source snapshots and unedited tracks bypass resampling. The interpolation regression checks smooth close knots, handle edits, rotation, nonuniform scale, hierarchy and a second import. The step regression covers constant transforms, native discontinuities, low frame rates and the explicit near-key time allowance. The group regression checks curve references when appending previously untracked bones and resampling cropped clips.

VIS texture offsets are scalar properties on the model owner, grouped by asset and visual variant. Their native attribute snapshots preserve unchanged decimal strings and detect conflicting XML edits. Older UV Empty objects migrate to these parameters on load, retaining pending edits and removing only empty owned collections. Attachment points remain bone-linked objects. The UV-parameter regression covers both native formats, migration, export fidelity, undo/redo, saved scenes and native XML conflicts.

Damage authoring writes native events to the model's retained Text document. The property groups project those entries and reference mesh objects so renamed parts export with their current names. `visuals/damage_authoring.py` owns selection validation, optional rigid binding, impact-point creation and the undoable operators. `visuals/damage_preview.py` owns derived debris objects in a collection identified by its model owner. They are excluded from exports and rebuilt or removed when the preview changes. Preview separation never changes source transforms, weights or native physics attributes. The damage-authoring regression checks both native formats, existing health thresholds, undo/redo, saved scenes and paired export/reimport.

Dependencies use explicit relative imports. `scripts/check.py` rejects circular imports during module initialization, checks the format boundary and verifies that development reloads follow dependency order. Shader-node callbacks import assignment and drawing operations when invoked, after registration is complete. Keep node `bl_idname` values and saved `hwek_ugx_*` property names stable.

## Source checks

Use Python 3.11 or newer. Blender supplies its own Python and runtime libraries.

```sh
python3 -m pip install -e ".[dev]"
python3 scripts/check.py
```

The checker reads syntax, Python names, package metadata, module boundaries and reload order. `--no-lint` omits the Python name check and needs no development dependency.

## Local tests

Tests stay outside the tracked repository and never run in GitHub Actions. Maintainers can retain `tests/` and `scripts/test.py` locally; both paths are ignored. Fresh clones do not include them. Existing local installations can continue using their runner:

```sh
python3 scripts/test.py --suite formats
python3 scripts/test.py --suite release
python3 scripts/test.py --suite blender --blender /path/to/blender
```

Game dumps and generated output stay outside the source tree. For format contracts and known editing limits, see [UGX and UAX](ugx-uax-workflow.md). `scripts/audit_native_assets.py` is an optional local asset audit; it requires your own extracted game files.

## Build and install

```sh
python3 scripts/package.py --output out/anders.zip
/path/to/blender --command extension validate out/anders.zip
python3 scripts/install_dev.py --repository /path/to/blender-development-repository
```

The packager reads the manifest's build paths, checks the compatibility version, includes the license, excludes tests and generated files, and writes a deterministic ZIP plus size/hash metadata. CMake uses this packager. The toolkit owns the Visual Studio utility project and stages the same root files and Python modules into an extension archive. Cutter terrain conversion ships separately with the toolkit and uses the installed Anders codecs.

Create a separate local development repository in Blender Preferences and point it at the directory supplied to `install_dev.py`. The installer copies the package there and replaces only that development copy. It does not change the source checkout or the normal installed extension.

For a loaded development package, use `importlib.reload(package_module)` in Blender's Python Console. This reload path removes the old menus, operators, handlers and timers, reloads modules in dependency order, and registers the addon again. Live custom shader-node RNA types are retained while their Python callbacks are refreshed, preserving existing material graphs. Changes to RNA schemas or node identifiers require a Blender restart.

Routine changes leave `src/metadata.py` and `blender_manifest.toml` versions unchanged. The manual [Release workflow](releases.md) updates both in a release-only child commit, validates the package and publishes its GitHub assets. Protected `main` retains development metadata.

Use short-lived branches and pull requests for changes. Both maintainers may merge after the required checks and resolved conversations; a second approval is not required. See [contribution guidance](../CONTRIBUTING.md).

The [official manual](https://docs.new-arcadia.net/anders) covers installation, asset editing, FBX conversion and toolkit terrain authoring. User documentation is maintained on that site rather than duplicated in this repository.
