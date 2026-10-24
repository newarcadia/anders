"""Check package syntax, format boundaries, reload order, and Python names."""
import argparse
import ast
from pathlib import Path

from package import extension_files, validate_version

PURE_MODULES = (
    'formats.binary', 'formats.ecf', 'formats.bdt', 'formats.granny', 'formats.matrix', 'formats.ufx', 'formats.xmb', 'formats.native_xml', 'formats.vis',
    'ugx.data', 'ugx.read', 'ugx.vertices', 'materials.data', 'animation.uax',
    'textures.header', 'textures.compression', 'textures.dds', 'textures.ddx', 'textures.paths',
)


def imports(tree, module):
    dependencies = set()
    external = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            external.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split('.')[:-1]
                keep = len(base) - node.level + 1
                name = '.'.join(base[:keep] + (node.module or '').split('.'))
                dependencies.add(name)
            else:
                external.add((node.module or '').split('.')[0])
        elif isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    external.update(alias.name.split('.')[0] for alias in child.names)
    return dependencies, external


def check_architecture(root, trees):
    modules = {'.'.join(path.relative_to(root / 'src').with_suffix('').parts): tree
               for path, tree in trees.items() if path.is_relative_to(root / 'src') and path.name != '__init__.py'}
    graph = {name: imports(tree, name) for name, tree in modules.items()}
    active, done = set(), set()

    def visit(name):
        if name in active:
            raise ValueError(f'Circular module import: {name}')
        if name in done:
            return
        active.add(name)
        for other in graph[name][0]:
            if other not in modules:
                raise ValueError(f'{name} imports a missing module: {other}')
            visit(other)
        active.remove(name)
        done.add(name)

    for name in modules:
        visit(name)
    for pure in PURE_MODULES:
        pending, seen = [pure], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            dependencies, external = graph[name]
            if external & {'bpy', 'bmesh', 'mathutils', 'bpy_extras'}:
                raise ValueError(f'Format module {pure} depends on Blender through {name}')
            pending.extend(dependencies)
    lifecycle = ast.literal_eval(next(node.value for node in modules['lifecycle'].body
                                     if isinstance(node, ast.Assign)))
    positions = {name: index for index, name in enumerate(lifecycle)}
    expected = set(modules) - {'lifecycle', 'compatibility'}
    if set(positions) != expected or len(positions) != len(lifecycle):
        raise ValueError('Reload order does not cover every implementation module exactly once')
    for name in expected:
        for dependency in graph[name][0]:
            if positions[dependency] >= positions[name]:
                raise ValueError(f'Reload order loads {name} before {dependency}')


def check_names(trees):
    try:
        from pyflakes.checker import Checker
    except ImportError:
        raise SystemExit('Install the development dependency: python -m pip install pyflakes==4.0.3')
    issues = []
    for path, original in trees.items():
        tree = ast.parse(ast.unparse(original))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Call):
                call = node.annotation.func
                if isinstance(call, ast.Name) and call.id.endswith('Property'):
                    # Blender evaluates RNA properties here, rather than Python type hints.
                    node.value = node.annotation
                    node.annotation = ast.Name(id='object', ctx=ast.Load())
        issues.extend(Checker(tree, filename=str(path)).messages)
    for issue in issues:
        print(issue)
    if issues:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-lint', action='store_true', help='Check syntax and module boundaries without development dependencies')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest, files = extension_files(root)
    validate_version(root, manifest)
    paths = [path for path in files if path.suffix == '.py']
    paths.extend((root / 'scripts').glob('*.py'))
    trees = {path: ast.parse(path.read_text(), filename=str(path)) for path in paths}
    check_architecture(root, trees)
    if not args.no_lint:
        check_names(trees)
    print('PACKAGE_SYNTAX_BOUNDARIES_AND_NAMES_OK', len(paths))


if __name__ == '__main__':
    main()
