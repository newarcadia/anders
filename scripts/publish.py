"""Reserve calendar versions and publish checked Anders snapshots on GitHub."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
from zoneinfo import ZoneInfo

from package import package

VERSION = re.compile(r'v([0-9]+)\.([0-9]+)\.([0-9]+)')
INSTALLED_VERSION_FLOOR = (2026, 10, 2301)
IDENTITY = 'devdotkuru'
EMAIL = '151408907+devdotkuru@users.noreply.github.com'


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def github_json(endpoint, missing_ok=False):
    result = subprocess.run(['gh', 'api', endpoint], capture_output=True, text=True)
    if result.returncode:
        if missing_ok and 'HTTP 404' in result.stderr:
            return None
        raise RuntimeError(f'GitHub request failed for {endpoint}: {result.stderr.strip()}')
    return json.loads(result.stdout)


def github_pages(endpoint):
    result = []
    page = 1
    while True:
        items = github_json(f'{endpoint}{"&" if "?" in endpoint else "?"}per_page=100&page={page}')
        result.extend(items)
        if len(items) < 100:
            return result
        page += 1


def release_version(tag):
    match = VERSION.fullmatch(tag)
    if match is None:
        raise ValueError(f'Unexpected release tag: {tag}')
    return tuple(map(int, match.groups()))


def source_for_tag(tag):
    commit = git('rev-parse', tag + '^{commit}')
    parents = git('rev-list', '--parents', '-n', '1', commit).split()[1:]
    if len(parents) != 1:
        raise ValueError(f'{tag} is not a release-only child commit')
    changed = set(git('diff-tree', '--no-commit-id', '--name-only', '-r', commit).splitlines())
    if not changed <= {'blender_manifest.toml', 'src/metadata.py'}:
        raise ValueError(f'{tag} changes source outside release metadata')
    manifest = tomllib.loads(git('show', commit + ':blender_manifest.toml'))
    if manifest['version'] != tag.removeprefix('v'):
        raise ValueError(f'{tag} does not match its manifest version')
    return parents[0], commit


def is_ancestor(first, last):
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', first, last])
    if result.returncode not in (0, 1):
        raise RuntimeError('Cannot establish release source ancestry')
    return result.returncode == 0


def published_state(repository, source):
    releases = github_pages(f'repos/{repository}/releases')
    published = [item for item in releases if not item['draft'] and not item['prerelease']]
    latest = max(published, key=lambda item: release_version(item['tag_name']), default=None)
    previous_source = None
    if latest:
        previous_source, _ = source_for_tag(latest['tag_name'])
        if not is_ancestor(previous_source, source):
            raise ValueError('The requested snapshot would regress or replace published source history')
    return releases, latest, previous_source


def apply_version(root, version):
    manifest = root / 'blender_manifest.toml'
    text, count = re.subn(r'^version = "[^"]+"$', f'version = "{version}"', manifest.read_text(), flags=re.M)
    if count != 1:
        raise ValueError('Cannot locate the extension version')
    manifest.write_text(text)
    metadata = root / 'src/metadata.py'
    values = ', '.join(version.split('.'))
    text, count = re.subn(r'"version": \([0-9, ]+\)', f'"version": ({values})', metadata.read_text())
    if count != 1:
        raise ValueError('Cannot locate the compatibility version')
    metadata.write_text(text)


def output_values(**values):
    with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
        for key, value in values.items():
            output.write(f'{key}={value}\n')


def allocate(repository, source, root):
    if git('rev-parse', 'HEAD') != source:
        raise ValueError('Allocation must start from the captured main snapshot')
    releases, latest, previous_source = published_state(repository, source)
    if previous_source == source:
        print('This source snapshot is already released')
        output_values(skip='true', source=source, commit='', version='')
        return
    tags = git('tag', '--list', 'v*').splitlines()
    # A tag reserves its version even if a job fails before creating the draft.
    pending = []
    for tag in tags:
        release_version(tag)
        released_source, commit = source_for_tag(tag)
        release = next((item for item in releases if item['tag_name'] == tag), None)
        if released_source == source and (release is None or release['draft']):
            pending.append((release_version(tag), tag, commit))
    floor = max([INSTALLED_VERSION_FLOOR] + [release_version(item['tag_name']) for item in releases if not item['draft']])
    if pending:
        _, tag, commit = max(pending)
        if release_version(tag) <= floor:
            raise ValueError('The reserved draft is older than the published version')
    else:
        now = datetime.now(ZoneInfo('Europe/Paris'))
        versions = {release_version(tag) for tag in tags}
        versions.update(release_version(item['tag_name']) for item in releases)
        version = next((f'{now.year}.{now.month}.{now.day * 100 + counter}' for counter in range(1, 100)
                        if (now.year, now.month, now.day * 100 + counter) > floor
                        and (now.year, now.month, now.day * 100 + counter) not in versions), None)
        if version is None:
            raise ValueError('No increasing Paris-calendar version is available today; release on a later date')
        apply_version(root, version)
        subprocess.run(['git', 'add', '--', 'blender_manifest.toml', 'src/metadata.py'], check=True)
        environment = dict(os.environ, GIT_AUTHOR_NAME=IDENTITY, GIT_AUTHOR_EMAIL=EMAIL,
                           GIT_COMMITTER_NAME=IDENTITY, GIT_COMMITTER_EMAIL=EMAIL)
        subprocess.run(['git', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', f'Release Anders {version}'],
                       env=environment, check=True)
        commit = git('rev-parse', 'HEAD')
        tag = 'v' + version
        subprocess.run(['git', '-c', 'tag.gpgsign=false', 'tag', tag, commit], check=True)
        subprocess.run(['git', '-c', 'credential.helper=', '-c', 'credential.helper=!gh auth git-credential',
                        'push', 'origin', f'refs/tags/{tag}'], check=True)
    notes = release_notes(repository, source, previous_source, tag)
    if not any(item['tag_name'] == tag for item in releases):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md') as file:
            file.write(notes); file.flush()
            subprocess.run(['gh', 'release', 'create', tag, '--repo', repository, '--verify-tag',
                            '--title', 'Anders ' + tag[1:], '--draft', '--notes-file', file.name], check=True)
    output_values(skip='false', source=source, commit=commit, version=tag[1:])


def release_notes(repository, source, previous_source, tag):
    archive = f'anders_ugx_tools-{tag[1:]}.zip'
    text = (f'Install **{archive}** through Blender Preferences > Get Extensions > Install from Disk.\n\n'
            f'For installation and editing guidance, see the [Anders manual](https://docs.new-arcadia.net/anders).\n')
    if previous_source:
        changes = []
        for pull in github_pages(f'repos/{repository}/pulls?state=closed&base=main&sort=updated&direction=desc'):
            commit = pull.get('merge_commit_sha')
            if pull.get('merged_at') and commit and is_ancestor(commit, source) and not is_ancestor(commit, previous_source):
                title = ' '.join(pull['title'].splitlines())
                changes.append(f'- {title} ([#{pull["number"]}]({pull["html_url"]}))')
        if changes:
            text += '\n## Changes\n\n' + '\n'.join(reversed(changes)) + '\n'
    text += (f'\nBlender update repository: https://github.com/{repository}/releases/latest/download/index.json\n'
             '\nInstall the extension ZIP above. The source archives are development snapshots.\n'
             f'\n<!-- anders-source: {source} -->\n')
    return text


def prepare_release(source, repository, output):
    manifest = tomllib.loads((source / 'blender_manifest.toml').read_text())
    version = manifest['version']
    tag = 'v' + version
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{manifest['id']}-{version}.zip"
    metadata = package(source, archive)
    entry = {key: value for key, value in manifest.items() if key != 'build'}
    entry.update({key: metadata[key] for key in ('archive_size', 'archive_hash')})
    entry['archive_url'] = f'https://github.com/{repository}/releases/download/{tag}/{archive.name}'
    index = output / 'index.json'
    index.write_text(json.dumps({'version': 'v1', 'blocklist': [], 'data': [entry]}, indent=2) + '\n')
    assets = [archive, archive.with_suffix('.zip.json'), index]
    checksums = output / 'SHA256SUMS'
    checksums.write_text(''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n' for path in assets))
    return tag, assets + [checksums]


def check_existing_assets(release, assets):
    expected = {path.name: path for path in assets}
    available = {item['name']: item for item in release['assets']}
    if not release['draft'] and not expected.keys() <= available.keys():
        raise ValueError('Published assets are incomplete and cannot be replaced')
    for name in expected.keys() & available.keys():
        result = subprocess.run(['gh', 'api', '-H', 'Accept: application/octet-stream', available[name]['url']],
                                capture_output=True, check=True)
        if result.stdout != expected[name].read_bytes():
            raise ValueError(f'{name} already contains different bytes; existing assets will not be overwritten')


def publish_release(root, repository, commit, output):
    if git('rev-parse', 'HEAD') != commit:
        raise ValueError('Publication must rebuild the checked release commit')
    tag, assets = prepare_release(root, repository, output)
    source, tagged_commit = source_for_tag(tag)
    if tagged_commit != commit:
        raise ValueError('The reserved tag changed after validation')
    releases, latest, previous_source = published_state(repository, source)
    release = next((item for item in releases if item['tag_name'] == tag), None)
    if release is None:
        raise ValueError('The reserved draft release is missing')
    check_existing_assets(release, assets)
    if not release['draft']:
        print(f'{tag} is already published with identical assets')
        return
    if latest and release_version(latest['tag_name']) >= release_version(tag):
        raise ValueError('Refusing to publish below the current release')
    available = {item['name'] for item in release['assets']}
    missing = [str(path) for path in assets if path.name not in available]
    if missing:
        subprocess.run(['gh', 'release', 'upload', tag, *missing, '--repo', repository], check=True)
    uploaded = github_json(f'repos/{repository}/releases/{release["id"]}')
    if not {path.name for path in assets} <= {item['name'] for item in uploaded['assets']}:
        raise ValueError('Draft upload is incomplete')
    check_existing_assets(uploaded, assets)
    # Main may advance while this snapshot builds. Only published ancestry matters.
    _, latest, previous_source = published_state(repository, source)
    if latest and release_version(latest['tag_name']) >= release_version(tag):
        raise ValueError('A newer version was published during upload')
    notes = output / 'release-notes.md'
    notes.write_text(release_notes(repository, source, previous_source, tag))
    subprocess.run(['gh', 'release', 'edit', tag, '--repo', repository, '--draft=false', '--latest',
                    '--notes-file', str(notes)], check=True)
    print(f'Published {tag} with {len(assets)} verified assets')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--commit', help='Captured main SHA for allocation, checked release SHA for publication')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allocate', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository):
        parser.error('Expected GitHub OWNER/REPO')
    root = Path(__file__).resolve().parents[1]
    if args.prepare_only:
        if not args.output:
            parser.error('--output is required')
        prepare_release(root, args.repository, args.output)
        return
    if (os.environ.get('GITHUB_ACTIONS') != 'true' or args.repository != 'newarcadia/anders'
            or os.environ.get('GITHUB_ACTOR') != IDENTITY or os.environ.get('GITHUB_TRIGGERING_ACTOR') != IDENTITY
            or os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch' or os.environ.get('GITHUB_REF') != 'refs/heads/main'):
        parser.error('Use the Release workflow on main from the devdotkuru account')
    if not args.commit or not re.fullmatch(r'[0-9a-f]{40}', args.commit):
        parser.error('A full commit SHA is required')
    if args.allocate:
        if os.environ.get('GITHUB_SHA') != args.commit:
            parser.error('Allocation requires the snapshot captured by the Release button')
        allocate(args.repository, args.commit, root)
    elif args.output:
        publish_release(root, args.repository, args.commit, args.output)
    else:
        parser.error('--output is required')


if __name__ == '__main__':
    main()
