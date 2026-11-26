import hashlib
import json
import ntpath
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp

import numpy as np

from .frame_dump import COLOR_FORMAT, UV_FORMATS, Draw, DumpError, Skinning, read_header


CORE = "\\APMIv1\\"
ACTOR_FIELDS = ("x", "y", "z", "facing_x", "facing_z")
MANIFEST = "apmi-tools.json"
FILE_PREFIX = "DISABLED_APMI_"
ATTRIBUTE_FORMATS = {name: (np.dtype(dtype), 2) for name, dtype in UV_FORMATS.items()}
ATTRIBUTE_FORMATS[COLOR_FORMAT] = np.dtype("u1"), 4
POSITION_FIELDS = {
    ("POSITION", 0, "R32G32B32_FLOAT", 0),
    ("NORMAL", 0, "R32G32B32_FLOAT", 12),
    ("TANGENT", 0, "R32G32B32A32_FLOAT", 24),
}
DDS_BLOCK_BYTES = {
    70: 8, 71: 8, 72: 8, 73: 16, 74: 16, 75: 16, 76: 16, 77: 16, 78: 16,
    79: 8, 80: 8, 81: 8, 82: 16, 83: 16, 84: 16, 94: 16, 95: 16, 96: 16,
    97: 16, 98: 16, 99: 16,
}
DDS_PIXEL_BYTES = {28: 4, 29: 4, 61: 1, 87: 4, 88: 4, 91: 4, 93: 4}


@dataclass(frozen=True)
class VertexField:
    semantic: str
    uv: int
    format: str
    offset: int


@dataclass
class Component:
    draw: Draw
    skinning: Skinning
    ib: str
    vb0: str
    passes: list[Draw]
    streams: dict[int, tuple[int, set[VertexField]]]
    textures: dict[str, tuple[set[Path], set[int]]]

    @property
    def label(self):
        return f"{self.draw.call:06d} ({self.ib})"


@dataclass
class MeshBuffers:
    vertices: int
    indices: int
    files: dict[str, bytes | bytearray]
    variants: dict | None = None
    variant_warning: str = ""


def get_component(dump, call):
    match = next(((draw, skin) for draw, skin in dump.draws if draw.call == call), None)
    if match is None:
        raise DumpError(f"Draw {call:06d} is missing from the source log")
    draw, skinning = match
    key = dump.component_key(draw, skinning)
    passes = [other for other, skin in dump.draws if dump.component_key(other, skin) == key]
    slots = dump.resources[call]
    ib, vb0 = slots["ib", "buf"].hash, slots["vb0", "buf"].hash
    if not all(re.fullmatch(r"[0-9a-f]{8}", value) for value in (ib, vb0)):
        raise DumpError(f"Draw {call:06d} has an invalid IB or VB0 hash")

    matched_calls = {entry.call for entry in passes}
    for other in dump.indexed_draws.values():
        if (other.first, other.count) != (draw.first, draw.count):
            continue
        resource = dump.resources.get(other.call, {}).get(("ib", "buf"))
        if resource and resource.hash == ib and other.call not in matched_calls:
            raise DumpError(f"IB {ib} and its draw range also match draw {other.call:06d}")

    streams, textures = {}, {}
    for entry in passes:
        slots = dump.resources[entry.call]
        vb0_header, elements = read_header(slots["vb0", "txt"].path)
        vb0_fields = {
            (element.semantic, element.number, element.format, element.offset)
            for element in elements if element.slot == 0
        }
        if int(vb0_header.get("stride", 0)) != 40 or not POSITION_FIELDS <= vb0_fields:
            raise DumpError(f"Draw {entry.call:06d} has an unsupported VB0 layout")

        slot_strides = {}
        for element in elements:
            slot = element.slot
            if slot == 0 or (slot == 3 and element.semantic == "TEXCOORD" and element.number == 5):
                continue
            layout = slots.get((f"vb{slot}", "txt"))
            if slot not in slot_strides:
                slot_strides[slot] = int(read_header(layout.path)[0].get("stride", 0)) if layout else 0
            stride = slot_strides[slot]
            if not stride:
                if layout and (f"vb{slot}", "buf") in slots:
                    continue
                raise DumpError(f"Draw {entry.call:06d} is missing metadata for VB{slot}")
            if slot not in {1, 4} or element.format not in ATTRIBUTE_FORMATS or not (
                element.semantic == "TEXCOORD" and element.number <= 2 or
                element.semantic == "COLOR" and element.number == 0
            ):
                raise DumpError(f"Draw {entry.call:06d} has an unsupported VB{slot} field")
            old_stride, fields = streams.setdefault(slot, (stride, set()))
            if old_stride != stride:
                raise DumpError(f"Draw {entry.call:06d} changes the stride of VB{slot}")
            fields.add(VertexField(element.semantic, element.number, element.format, element.offset))

        for (slot, extension), resource in slots.items():
            if slot.startswith("ps-t") and extension == "dds" and re.fullmatch(r"[0-9a-f]{8}", resource.hash):
                paths, bindings = textures.setdefault(resource.hash, (set(), set()))
                paths.add(resource.path)
                bindings.add(int(slot[4:]))

    if not any(field.semantic == "TEXCOORD" and field.uv == 0 for _, fields in streams.values() for field in fields):
        raise DumpError(f"Draw {call:06d} has no surface UVs")
    return Component(draw, skinning, ib, vb0, passes, streams, textures)


def dds_format(path):
    with path.open("rb") as stream:
        header = stream.read(148)
    if len(header) != 148 or header[:4] != b"DDS " or header[84:88] != b"DX10":
        raise DumpError(f"{path.name} must be a DX10 DDS texture")
    size, _, height, width = struct.unpack_from("<4I", header, 4)
    mip_count = max(1, struct.unpack_from("<I", header, 28)[0])
    fmt, dimension, flags, array_size, _ = struct.unpack_from("<5I", header, 128)
    if size != 124 or dimension != 3 or array_size != 1 or flags & 4 or not width or not height:
        raise DumpError(f"{path.name} must be a single 2D DDS texture")
    if mip_count > max(width, height).bit_length():
        raise DumpError(f"{path.name} has an invalid DDS mip count")
    return fmt, width, height, mip_count


def replacement_textures(folder, components):
    if folder is None:
        return {}, []
    if not folder.is_dir():
        raise DumpError("The DDS replacement folder does not exist")
    sources = {}
    for component in components:
        for texture_hash, (paths, _) in component.textures.items():
            sources.setdefault(texture_hash, set()).update(paths)

    replacements, ignored = {}, []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() != ".dds" or not path.is_file():
            continue
        texture_hash = path.stem.lower()
        if texture_hash not in sources:
            ignored.append(path.name)
            continue
        if texture_hash in replacements:
            raise DumpError(f"More than one DDS matches {texture_hash}")
        fmt, width, height, mips = dds_format(path)
        if any(dds_format(original)[0] != fmt for original in sources[texture_hash]):
            raise DumpError(f"{path.name} does not match the original DDS format")
        block_bytes, pixel_bytes = DDS_BLOCK_BYTES.get(fmt), DDS_PIXEL_BYTES.get(fmt)
        if not (block_bytes or pixel_bytes):
            raise DumpError(f"{path.name} uses an unsupported DDS format ({fmt})")
        required = 148
        for _ in range(mips):
            if block_bytes:
                required += ((width + 3) // 4) * ((height + 3) // 4) * block_bytes
            else:
                required += width * height * pixel_bytes
            width, height = max(1, width // 2), max(1, height // 2)
        if path.stat().st_size < required:
            raise DumpError(f"{path.name} has an incomplete DDS payload")
        replacements[texture_hash] = path
    return replacements, ignored


def profile_condition(profile, prefix):
    return (
        f"${CORE}{prefix}_blend_hi == {int(profile.blend_hash[:4], 16)}"
        f" && ${CORE}{prefix}_blend_lo == {int(profile.blend_hash[4:], 16)}"
        f" && ${CORE}{prefix}_format == {profile.bones * 8 + profile.influences}"
    )


def donor_sections(donor, number, collect):
    ib, first, count, base = donor.component
    name = f"Donor{number}"
    actor = f"${CORE}actor_key"
    commands = [
        f"[TextureOverride{name}]",
        f"hash = {ib}",
        f"match_first_index = {first}",
        f"match_index_count = {count}",
        f"match_first_vertex = {base & 0xffffffff}",
        "match_priority = -100",
        "if $mod_id > 0 && (DRAW_TYPE == 2 || (DRAW_TYPE == 4 && INSTANCE_COUNT == 1))",
        f"    run = CommandList{CORE}ReadSkinActor",
        f"    if {actor} > 0 && ({profile_condition(donor.profile, 'actor')})",
        f"        run = CommandList{CORE}RecordSkinActor",
    ]
    sections = []
    for parity in (0, 1):
        pool = f"Pool{name}{parity}"
        owner = f"Pool{name}Output{parity}"
        fields = [(f"Pool{name}{field}{parity}", f"${CORE}actor_{field}") for field in ACTOR_FIELDS]
        palette_flags = ("\npool_element_type_switch_reset = 0\ntype = StructuredBuffer\n"
                         "stride = 64\nbind_flags = shader_resource") if collect else ""
        for resource in (pool, owner, *(field for field, _ in fields)):
            sections.append(f"[{resource}]\npool_size = 64\npool_index_type = fifo"
                            + (palette_flags if resource == pool else ""))
        conflicts = [f"${owner}[{actor}] != @vb0",
                     *(f"${field}[{actor}] != {value}" for field, value in fields)]
        commands.extend([
            f"        if FRAME_NUMBER % 2 == {parity}",
            f"            if ${pool}[{actor}] == FRAME_NUMBER + 1",
            f"                if {' || '.join(conflicts)}",
            f"                    ${pool}[{actor}] = -(FRAME_NUMBER + 1)",
            "                endif",
            f"            elif ${pool}[{actor}] != -(FRAME_NUMBER + 1)",
        ])
        indent = "                "
        if collect:
            commands.extend([
                f"{indent}{pool}[{actor}] = copy Pool{CORE}BoneMatrices[@vb0]",
                f"{indent}if @{pool}[{actor}]",
            ])
            indent += "    "
        commands.extend([f"{indent}${owner}[{actor}] = @vb0",
                         *(f"{indent}${field}[{actor}] = {value}" for field, value in fields),
                         f"{indent}${pool}[{actor}] = FRAME_NUMBER + 1"])
        if collect:
            commands.append("                endif")
        commands.extend(["            endif", "        endif"])
    return ["\n".join([*commands, "    endif", "endif"]), *sections]


def merged_selection(name, variant, receiver, donor_ids):
    actor = f"${CORE}snapshot_actor"
    receiver = f"Donor{donor_ids[receiver]}"
    commands = [
        f"    if !${CORE}variant_is_previous",
        f"        run = CommandList{CORE}ReadPreviousSkinActor",
        f"        if {actor} > 0",
    ]
    names = [(f"Donor{donor_ids[donor]}", table) for donor, table in variant.donors]
    for parity in (0, 1):
        def snapshot(donor, field):
            return f"$Pool{donor}{field}{parity}[{actor}]"

        current_actor = " && ".join(
            f"{snapshot(receiver, field)} == ${CORE}actor_{field}" for field in ACTOR_FIELDS)
        receiver_checks = [
            f"{snapshot(receiver, '')} == FRAME_NUMBER",
            f"{snapshot(receiver, 'Output')} == ${CORE}snapshot_output",
            f"(${CORE}snapshot_tracks_motion || ({current_actor}))",
        ]
        commands.extend([
            f"            if (FRAME_NUMBER - 1) % 2 == {parity}",
            f"                local $ready = {' && '.join(receiver_checks)}",
        ])
        for donor, _ in names:
            checks = [f"{snapshot(donor, '')} == FRAME_NUMBER", f"@Pool{donor}{parity}[{actor}]"]
            if donor != receiver:
                checks.extend(f"{snapshot(donor, field)} == {snapshot(receiver, field)}" for field in ACTOR_FIELDS)
            commands.append(f"                $ready = $ready && {' && '.join(checks)}")
        commands.extend([
            "                if $ready",
            f"                    Resource{CORE}GatherDestination = ref Resource{name}MergedPalette",
        ])
        for donor, table in names:
            commands.extend([
                f"                    Resource{CORE}GatherSource = ref Pool{donor}{parity}[{actor}]",
                f"                    Resource{CORE}GatherMap = ref Resource{name}{table}",
                f"                    run = CommandList{CORE}GatherSkinBones",
            ])
        commands.extend([
            f"                    Resource{CORE}VariantPalette = ref Resource{name}MergedPalette",
            f"                    Resource{CORE}VariantBlend = ref Resource{name}{variant.blend}",
            "                endif",
            "            endif",
        ])
    return [*commands, "        endif", "    endif"]


def render_ini(mod_name, parts, replacements):
    from .skinning_variants import BoneSource, draw_key

    namespace = "APMITools_" + hashlib.sha256(mod_name.casefold().encode("utf-8")).hexdigest()[:16]
    if not parts:
        return f"namespace = {namespace}\n"

    display_name = mod_name.replace('"', "'")
    donors = {}
    variant_hashes = set()
    for component, buffers in parts:
        for profile, variant in (buffers.variants or {}).items():
            variant_hashes.add(profile.blend_hash)
            if variant.donors:
                receiver = BoneSource(draw_key(component.ib, component.draw), profile, 0)
                donors.setdefault(receiver, False)
            for donor, _ in variant.donors:
                donors[donor] = True
                variant_hashes.add(donor.profile.blend_hash)
    donor_ids = {donor: number for number, donor in enumerate(donors)}
    required_version = "0.31" if any(buffers.variants is not None for _, buffers in parts) else "0.22"
    sections = [
        f"namespace = {namespace}",
        "[Constants]\nglobal $mod_id = 0\nglobal $replace_textures = 0",
        f'[ResourceModName]\ntype = Buffer\ndata = "{display_name}"',
        "\n".join([
            "[Present]",
            "if $mod_id == 0",
            f"    ${CORE}required_version = {required_version}",
            f"    Resource{CORE}ModName = ref ResourceModName",
            f"    run = CommandList{CORE}RegisterMod",
            f"    $mod_id = ${CORE}mod_id",
            "endif",
        ]),
    ]
    for blend_hash in sorted(variant_hashes):
        sections.append("\n".join([
            f"[TextureOverrideNativeBlend{blend_hash}]",
            f"hash = {blend_hash}",
            "match_priority = 0",
            f"if ${CORE}identify_native_blend",
            f"    ${CORE}native_blend_hi = {int(blend_hash[:4], 16)}",
            f"    ${CORE}native_blend_lo = {int(blend_hash[4:], 16)}",
            "endif",
        ]))
    for donor, collect in donors.items():
        sections.extend(donor_sections(donor, donor_ids[donor], collect))
    texture_slots = set()
    for n, (component, buffers) in enumerate(parts):
        name = f"Part{n:02d}"
        draw = component.draw
        draw_types = {4 if entry.instances is not None else 2 for entry in component.passes}
        allow = " || ".join(f"DRAW_TYPE == {kind}" + (" && INSTANCE_COUNT == 1" if kind == 4 else "")
                            for kind in sorted(draw_types))
        skin_mesh = "Position" in buffers.files
        replace_indices = "IB" in buffers.files
        override = [
            f"[TextureOverride{name}]",
            f"hash = {component.ib}",
            f"match_first_index = {draw.first}",
            f"match_index_count = {draw.count}",
        ]
        if not skin_mesh or buffers.variants is not None:
            override.append(f"match_first_vertex = {draw.base & 0xffffffff}")
        override.append(f"if $mod_id > 0 && ({allow})")
        if skin_mesh:
            if buffers.variants is not None:
                override.extend([
                    f"    ${CORE}variant_mode = 1",
                    f"    CommandList{CORE}Callback_SelectSkinVariant = ref CommandListSelect{name}",
                ])
                selector = [f"[CommandListSelect{name}]"]
                for profile, variant in buffers.variants.items():
                    selector.append("if " + profile_condition(profile, "variant"))
                    if variant.donors:
                        sections.append(
                            f"[Resource{name}MergedPalette]\ntype = StructuredBuffer\nstride = 64\n"
                            f"array = {variant.bone_count}\nbind_flags = shader_resource unordered_access"
                        )
                        receiver = BoneSource(draw_key(component.ib, draw), profile, 0)
                        selector.extend(merged_selection(name, variant, receiver, donor_ids))
                    else:
                        selector.append(f"    Resource{CORE}VariantBlend = ref Resource{name}{variant.blend}")
                    selector.append("endif")
                sections.append("\n".join(selector))
            override.extend([
                f"    ${CORE}component_id = {n}",
                f"    ${CORE}gpu_posed = 1",
                f"    Resource{CORE}Input_Position = ref Resource{name}Position",
                f"    Resource{CORE}Input_Blend = ref Resource{name}Blend",
                f"    CommandList{CORE}Callback_Component_DrawCustom = ref CommandListDraw{name}",
                f"    run = CommandList{CORE}Component_DrawInstances",
            ])
        else:
            override.extend(["    handling = skip", f"    run = CommandListDraw{name}"])
        sections.append("\n".join([*override, "endif"]))

        commands = [f"[CommandListDraw{name}]"]
        if replace_indices:
            commands.append(f"ib = ref Resource{name}IB")
        if skin_mesh:
            commands.append(f"vb0 = ref Resource{CORE}Position")
            commands.extend(f"vb{slot} = ref Resource{name}VB{slot}" for slot in sorted(component.streams))
        bindings = sorted({slot for texture_hash, (_, slots) in component.textures.items()
                           if texture_hash in replacements for slot in slots})
        texture_slots.update(bindings)
        if bindings:
            commands.extend(f"ResourceSavedPST{slot} = ref ps-t{slot}" for slot in bindings)
            commands.append("$replace_textures = 1")
            commands.extend(f"checktextureoverride = ps-t{slot}" for slot in bindings)
            commands.append("$replace_textures = 0")
        first = 0 if replace_indices else draw.first
        base = 0 if skin_mesh else draw.base
        indexed = f"drawindexed = {buffers.indices}, {first}, {base}"
        instanced = f"drawindexedinstanced = {buffers.indices}, INSTANCE_COUNT, {first}, {base}, FIRST_INSTANCE"
        if draw_types == {4}:
            commands.append(instanced)
        elif draw_types == {2, 4}:
            commands.extend(["if DRAW_TYPE == 4", f"    {instanced}", "else", f"    {indexed}", "endif"])
        else:
            commands.append(indexed)
        commands.extend(f"ps-t{slot} = ref ResourceSavedPST{slot}" for slot in bindings)
        commands.extend(f"ResourceSavedPST{slot} = null" for slot in bindings)
        sections.append("\n".join(commands))

        for buffer, config in {
            "IB": "type = Buffer\nformat = DXGI_FORMAT_R32_UINT\nbind_flags = index_buffer",
            "Position": "type = ByteAddressBuffer\nformat = DXGI_FORMAT_R32_TYPELESS\nbind_flags = shader_resource",
            "Blend": "type = ByteAddressBuffer\nformat = DXGI_FORMAT_R32_TYPELESS\nbind_flags = shader_resource",
            "BlendField": "type = ByteAddressBuffer\nformat = DXGI_FORMAT_R32_TYPELESS\nbind_flags = shader_resource",
            **{key: "type = StructuredBuffer\nstride = 8\nbind_flags = shader_resource"
               for key in buffers.files if key.startswith("BoneMap")},
            **{f"VB{slot}": f"type = Buffer\nstride = {stride}\nbind_flags = vertex_buffer"
               for slot, (stride, _) in component.streams.items()},
        }.items():
            if buffer in buffers.files:
                sections.append(f"[Resource{name}{buffer}]\n{config}\nfilename = Buffers/{name}{buffer}.buf")

    sections.extend(f"[ResourceSavedPST{slot}]" for slot in sorted(texture_slots))
    for texture_hash in sorted(replacements):
        sections.extend([
            f"[ResourceTexture{texture_hash}]\nfilename = Textures/{texture_hash}.dds",
            f"[TextureOverrideTexture{texture_hash}]\nhash = {texture_hash}\n"
            f"if $replace_textures\n    this = ResourceTexture{texture_hash}\nendif",
        ])
    return "\n\n".join(sections) + "\n"


def valid_mod_name(name):
    return bool(name) and not (
        name.startswith((".", FILE_PREFIX)) or name.endswith((" ", ".")) or
        re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or ntpath.isreserved(name)
    )


def output_target(filepath):
    if not filepath or not Path(filepath).name:
        raise DumpError("Choose a destination and enter a mod folder name")
    path = Path(filepath)
    name = path.name
    if not valid_mod_name(name):
        raise DumpError("The mod folder name is invalid")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise DumpError("The mod destination must be a folder")
    return parent / name


def verify_owned(target):
    manifest = target / MANIFEST
    try:
        if target.is_symlink() or target.is_junction() or manifest.is_symlink():
            raise ValueError
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data["generator"] != "APMI-Tools" or data["format"] != 1:
            raise ValueError
        paths = set(data["files"]) | {MANIFEST}
        directories = {p.as_posix() for name in paths for p in PurePosixPath(name).parents}
        for path in target.rglob("*"):
            relative = path.relative_to(target).as_posix()
            if path.is_symlink() or path.is_junction() or relative not in (directories if path.is_dir() else paths):
                raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise DumpError(f"{target.name} contains files outside an APMI-Tools export") from error


def write_mod(target, parts, replacements, overwrite):
    lock_id = hashlib.sha256(target.name.casefold().encode("utf-8")).hexdigest()[:16]
    lock = target.parent / f"{FILE_PREFIX}{lock_id}.lock"
    try:
        with lock.open("x", encoding="ascii"):
            pass
    except FileExistsError as error:
        raise DumpError(f"An export for {target.name} is already in progress") from error

    stage = None
    retain_stage = False
    try:
        exists = target.exists() or target.is_symlink()
        if exists:
            if not overwrite:
                raise DumpError(f"{target.name} already exists. Enable Overwrite APMI Export or choose another folder")
            verify_owned(target)
        stage = Path(mkdtemp(prefix=FILE_PREFIX, dir=target.parent))
        new = stage / "new"
        new.mkdir()
        (new / "Buffers").mkdir()
        files = ["mod.ini"]
        for n, (_, buffers) in enumerate(parts):
            for name, data in buffers.files.items():
                relative = f"Buffers/Part{n:02d}{name}.buf"
                (new / relative).write_bytes(data)
                files.append(relative)
        if replacements:
            (new / "Textures").mkdir()
            for texture_hash, source in sorted(replacements.items()):
                relative = f"Textures/{texture_hash}.dds"
                shutil.copyfile(source, new / relative)
                files.append(relative)
        (new / "mod.ini").write_text(render_ini(target.name, parts, replacements), encoding="utf-8")
        (new / MANIFEST).write_text(json.dumps({
            "generator": "APMI-Tools", "format": 1, "files": sorted(files),
        }, indent=2) + "\n", encoding="utf-8")

        backup = stage / "original"
        if exists:
            target.rename(backup)
        try:
            new.rename(target)
        except OSError:
            if exists:
                try:
                    backup.rename(target)
                except OSError as error:
                    retain_stage = True
                    raise DumpError(f"Could not restore the previous export. It is at {backup}") from error
            raise
    finally:
        try:
            if stage and not retain_stage and stage.resolve() == stage and stage.name.startswith(FILE_PREFIX):
                shutil.rmtree(stage)
        finally:
            lock.unlink()
