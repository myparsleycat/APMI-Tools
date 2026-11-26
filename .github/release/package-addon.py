import ast
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def versioned_init(source, version):
    metadata = next(
        (
            statement.value
            for statement in ast.parse(source).body
            if isinstance(statement, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "bl_info" for target in statement.targets)
        ),
        None,
    )
    if not isinstance(metadata, ast.Dict):
        raise ValueError("Expected a bl_info dictionary in __init__.py")

    current_version = next(
        (value for key, value in zip(metadata.keys, metadata.values) if isinstance(key, ast.Constant) and key.value == "version"),
        None,
    )
    if not isinstance(current_version, ast.Tuple):
        raise ValueError("Expected a bl_info version tuple in __init__.py")

    lines = source.splitlines(keepends=True)
    start = current_version.lineno - 1
    end = current_version.end_lineno - 1
    lines[start : end + 1] = [
        lines[start][:current_version.col_offset]
        + str(version).encode("ascii")
        + lines[end][current_version.end_col_offset:]
    ]
    return b"".join(lines)


def main(version, output):
    version_tuple = tuple(map(int, version.split(".")))
    addon_files = (
        "__init__.py",
        "blender_export.py",
        "frame_dump.py",
        "mod_export.py",
        "options.py",
        "sidebar.py",
        "skinning_variants.py",
        "LICENSE",
        "UPSTREAM.md",
    )
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name in addon_files:
            source = Path(name).read_bytes()
            if name == "__init__.py":
                source = versioned_init(source, version_tuple)
            archive.writestr(f"apmi_tools/{name}", source)


if __name__ == "__main__":
    main(*sys.argv[1:])
