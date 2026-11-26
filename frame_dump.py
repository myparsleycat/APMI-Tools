import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# The value is the skin profile tag: the fixed influence count, or VARIABLE_INFLUENCES for a
# blend buffer with a per-vertex offset table. Core stores bones * 8 + tag, so a tag must be below 8.
VARIABLE_INFLUENCES = 3
MAX_INFLUENCES = 8
SKINNING_SHADERS = {
    "3b4f1b1749cd3030": 1,
    "10749e66a6e6de89": 2,
    "09d91b2eee7e232f": 4,
    "38cddb8c0126726a": VARIABLE_INFLUENCES,
}
POSITION_STRIDE = 40
CALL = re.compile(r"^(\d+) (\w+)\(([^)]*)\)")
ARGUMENT = re.compile(r"(\w+):(-?\d+)(?=,|$)")
BINDING = re.compile(r"^\s+(\d+): .*?\bresource=(0x[0-9a-fA-F]+)")
REDIRECT = re.compile(r"^\d+ 3DMigoto .*?Dumping .*? (.+?) -> (.+)$")
RESOURCE_NAME = re.compile(
    r"^(\d+)-(ib|vb\d+|u0|cs-t[0-2]|vs-cb1|ps-t\d+)=([^-]+)(-.*)?\.(buf|txt|dds)$",
    re.IGNORECASE,
)
INDEX_FORMATS = {"R16_UINT": "<u2", "R32_UINT": "<u4"}
UV_FORMATS = {"R16G16_FLOAT": "<f2", "R32G32_FLOAT": "<f4"}
COLOR_FORMAT = "R8G8B8A8_UNORM"


class DumpError(ValueError):
    pass


def frame_log(path):
    path = Path(path)
    log = path / "log.txt" if path.is_dir() else path
    if not log.is_file() or log.name.lower() != "log.txt":
        raise DumpError(f"No Frame Analysis log.txt in {path}")
    return log.resolve()


@dataclass(frozen=True)
class Resource:
    hash: str
    path: Path
    shaders: str


@dataclass(frozen=True)
class Skinning:
    call: int
    influence_count: int
    rest: Resource | None
    blend: Resource | None
    palette: Resource | None


@dataclass(frozen=True)
class Draw:
    call: int
    first: int
    count: int
    base: int
    instances: int | None = None


@dataclass
class Part:
    draw: Draw
    skinning: Skinning
    index_hash: str
    vertex_hash: str
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    vertex_ids: np.ndarray
    triangles: np.ndarray
    uvs: dict[int, np.ndarray]
    color: np.ndarray | None
    weights: np.ndarray | None
    bones: np.ndarray | None
    textures: dict[int, Resource] = field(default_factory=dict)


@dataclass(frozen=True)
class Element:
    semantic: str
    number: int
    format: str
    slot: int
    offset: int


def read_header(path):
    header, elements = {}, []
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            name, sep, value = line.strip().partition(":")
            if not sep:
                continue
            if name in {"vertex-data", "instance-data"}:
                break
            if name.startswith("element["):
                elements.append({})
            elif elements:
                elements[-1][name] = value.strip()
            else:
                header[name] = value.strip()
            if name == "format" and "index count" in header:
                break
    return header, [
        Element(
            entry["SemanticName"], int(entry["SemanticIndex"]),
            entry["Format"].removeprefix("DXGI_FORMAT_"), int(entry["InputSlot"]),
            int(entry["AlignedByteOffset"]),
        )
        for entry in elements if entry.get("InputSlotClass") == "per-vertex"
    ]


def rows(path, stride, offset, dtype, width, ids):
    data = path.read_bytes()
    dtype = np.dtype(dtype)
    if offset < 0 or offset + dtype.itemsize * width > len(data):
        raise DumpError(f"Invalid buffer offset in {path.name}")
    last = offset + (int(ids[-1]) * stride) + dtype.itemsize * width
    if last > len(data):
        raise DumpError(f"Vertex indices exceed {path.name}")
    view = np.ndarray(
        (int(ids[-1]) + 1, width), dtype=dtype, buffer=data,
        offset=offset, strides=(stride, dtype.itemsize),
    )
    return view[ids]


def variable_rows(path, vertices, ids):
    # Layout: (vertices + 1) uint32 offsets in uint32 units, then one uint32 per influence
    # holding the palette index in the low 16 bits and a UNORM16 weight in the high 16 bits.
    data = np.fromfile(path, dtype="<u4")
    if len(data) <= vertices:
        raise DumpError("Unexpected cs-t1 blend buffer size")
    offsets = data[:vertices + 1].astype(np.int64)
    if offsets[0] != vertices + 1:
        raise DumpError("cs-t1 vertex count does not match cs-t0")
    spans = np.diff(offsets)
    if offsets[-1] != len(data) or np.any(spans < 1) or np.any(spans > MAX_INFLUENCES):
        raise DumpError("Invalid cs-t1 influence offsets")
    if int(ids[-1]) >= vertices:
        raise DumpError(f"Vertex indices exceed {path.name}")
    slots = np.arange(MAX_INFLUENCES)
    active = slots < spans[ids][:, None]
    packed = np.zeros((len(ids), MAX_INFLUENCES), dtype=np.uint32)
    packed[active] = data[(offsets[ids][:, None] + slots)[active]]
    weights = (packed >> 16).astype(np.float32) / np.float32(65535)
    return weights, packed & 0xFFFF


def parse_filter(value):
    calls, hashes = set(), set()
    for token in re.split(r"[,;\s]+", value.strip().lower()):
        if not token:
            continue
        if re.fullmatch(r"[0-9a-f]{8}", token):
            hashes.add(token)
        elif re.fullmatch(r"[0-9]{1,6}", token):
            calls.add(int(token))
        else:
            raise DumpError(f"Invalid draw number or index-buffer hash: {token}")
    return calls, hashes


class FrameDump:
    def __init__(self, log):
        self.root = log.parent
        self.resources = {}
        self.draws = []
        self.unsupported = []
        self.indexed_draws = {}
        self.read_log(log)

    def read_log(self, log):
        names, redirects, events = {}, {}, []
        bound = {"vb0": None, "u0": None}
        binding_slot = None
        with log.open(encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                line = line.rstrip()
                if line.startswith(" "):
                    if binding_slot and (match := BINDING.match(line)) and match[1] == "0":
                        bound[binding_slot] = int(match[2], 16) or None
                    continue
                binding_slot = None
                if match := REDIRECT.match(line):
                    source, target = (item.replace("\\", "/").rsplit("/", 1)[-1] for item in match.groups())
                    redirects[source] = target
                    names.setdefault(source, self.root / source)
                    continue
                if not (match := CALL.match(line)):
                    continue
                call, action, payload = match.groups()
                if action in {"IASetVertexBuffers", "CSSetUnorderedAccessViews"}:
                    args = dict(ARGUMENT.findall(payload))
                    if args.get("StartSlot") == "0" and args.get("NumBuffers", args.get("NumUAVs")) != "0":
                        binding_slot = "vb0" if action == "IASetVertexBuffers" else "u0"
                        bound[binding_slot] = None
                elif action == "ClearState":
                    bound = dict.fromkeys(bound)
                elif action == "Dispatch":
                    events.append((int(call), bound["u0"], None))
                elif action in {"DrawIndexed", "DrawIndexedInstanced"}:
                    args = {key: int(value) for key, value in ARGUMENT.findall(payload)}
                    count = args.get("IndexCount", args.get("IndexCountPerInstance", 0))
                    if count > 0:
                        draw = Draw(
                            int(call), args["StartIndexLocation"], count, args["BaseVertexLocation"],
                            args.get("InstanceCount"),
                        )
                        self.indexed_draws[draw.call] = draw
                        if draw.instances in {None, 1}:
                            events.append((draw.call, bound["vb0"], draw))

        for path in self.root.iterdir():
            names.setdefault(path.name, path)
        for name, path in sorted(names.items()):
            if not (match := RESOURCE_NAME.fullmatch(name)):
                continue
            call, slot, resource_hash, shaders, ext = match.groups()
            if path.is_file():
                path = path.resolve()
            elif name in redirects and (target := self.root / "deduped" / redirects[name]).is_file():
                path = target
            else:
                continue
            slots = self.resources.setdefault(int(call), {})
            slots.setdefault((slot.lower(), ext.lower()), Resource(resource_hash.lower(), path, shaders or ""))

        by_address, by_file = {}, {}
        for call, address, draw in events:
            slots = self.resources.get(call, {})
            if draw is None:
                output = slots.get(("u0", "buf"))
                shader = re.search(r"-cs=([0-9a-f]{16})$", output.shaders) if output else None
                if address:
                    by_address.pop(address, None)
                if not shader:
                    continue
                if shader[1] in SKINNING_SHADERS:
                    skinning = Skinning(call, SKINNING_SHADERS[shader[1]], *(
                        slots.get((slot, "buf")) for slot in ("cs-t0", "cs-t1", "cs-t2")
                    ))
                    by_file[output.path] = skinning
                elif ("cs-t2", "buf") in slots:
                    # A bone palette marks an unknown skinning shader rather than another compute pass.
                    skinning = shader[1]
                    by_file.setdefault(output.path, skinning)
                else:
                    continue
                if address:
                    by_address[address] = skinning
            elif (vertex := slots.get(("vb0", "buf"))) and ("ib", "buf") in slots:
                skinning = by_address.get(address) if address else by_file.get(vertex.path)
                if isinstance(skinning, str):
                    self.unsupported.append((draw, skinning))
                elif skinning:
                    self.draws.append((draw, skinning))

    def decode_part(self, draw, skinning, pre_skinning, import_weights):
        slots = self.resources[draw.call]

        def require(slot, ext):
            if (resource := slots.get((slot, ext))) is None:
                raise DumpError(f"Missing {slot}.{ext}")
            return resource

        ib = require("ib", "buf")
        ib_header, _ = read_header(require("ib", "txt").path)
        dtype = INDEX_FORMATS.get(ib_header.get("format", "").removeprefix("DXGI_FORMAT_"))
        if ib_header.get("topology") != "trianglelist" or not dtype or draw.count % 3:
            raise DumpError("Unsupported index format or topology")
        raw = ib.path.read_bytes()
        stride = np.dtype(dtype).itemsize
        start = int(ib_header.get("byte offset", 0)) + draw.first * stride
        if start < 0 or start + draw.count * stride > len(raw):
            raise DumpError("Index draw range exceeds the buffer")
        indices = np.frombuffer(raw, dtype=dtype, count=draw.count, offset=start).astype(np.int64)
        ids, inverse = np.unique(indices + draw.base, return_inverse=True)
        if ids[0] < 0 or ids[-1] > np.iinfo(np.int32).max:
            raise DumpError("Vertex index is out of range")

        vb = require("vb0", "buf")
        header, elements = read_header(require("vb0", "txt").path)
        if int(header.get("stride", 0)) != POSITION_STRIDE or Element(
            "POSITION", 0, "R32G32B32_FLOAT", 0, 0
        ) not in elements:
            raise DumpError("Unsupported APMI position layout")
        offset = int(header.get("byte offset", 0))
        source = vb
        if pre_skinning:
            source = skinning.rest
            if source is None or source.path.stat().st_size + offset != vb.path.stat().st_size:
                raise DumpError("Missing or incompatible cs-t0 position buffer")
        values = rows(source.path, POSITION_STRIDE, 0 if pre_skinning else offset, "<f4", 10, ids)
        if not np.isfinite(values).all():
            raise DumpError("Non-finite position, normal or tangent values")

        # The layouts contain placeholder semantics aliased to POSITION and a stride-zero color buffer.
        uvs, color = {}, None
        headers = {0: header}
        for element in elements:
            is_uv = element.semantic == "TEXCOORD" and element.number <= 2 and element.format in UV_FORMATS
            is_color = element.semantic == "COLOR" and element.number == 0 and element.format == COLOR_FORMAT
            if not (is_uv or is_color):
                continue
            slot = element.slot
            resource = slots.get((f"vb{slot}", "buf"))
            layout = slots.get((f"vb{slot}", "txt"))
            if not resource or not layout:
                continue
            if slot not in headers:
                headers[slot], _ = read_header(layout.path)
            slot_header = headers[slot]
            slot_stride = int(slot_header.get("stride", 0))
            value_type, width = (UV_FORMATS[element.format], 2) if is_uv else ("u1", 4)
            if slot_stride < element.offset + np.dtype(value_type).itemsize * width:
                continue
            slot_offset = int(slot_header.get("byte offset", 0)) + element.offset
            attribute = rows(resource.path, slot_stride, slot_offset, value_type, width, ids).astype(np.float32)
            if is_uv and np.isfinite(attribute).all():
                attribute[:, 1] = 1.0 - attribute[:, 1]
                uvs[element.number] = attribute
            elif is_color:
                color = attribute / 255.0

        weights, bones = self.decode_weights(skinning, ids) if import_weights else (None, None)
        return Part(
            draw, skinning, ib.hash, vb.hash, values[:, :3], values[:, 3:6],
            values[:, 6:10], ids.astype(np.int32), inverse.astype(np.int32).reshape(-1, 3),
            uvs, color, weights, bones,
        )

    @staticmethod
    def decode_weights(skinning, ids):
        blend, rest, palette = skinning.blend, skinning.rest, skinning.palette
        if not blend or not rest:
            raise DumpError("Missing cs-t0 or cs-t1 buffer for vertex weights")
        count = skinning.influence_count
        positions_size = rest.path.stat().st_size
        if positions_size % POSITION_STRIDE:
            raise DumpError("Unexpected cs-t0 position buffer size")
        if count == VARIABLE_INFLUENCES:
            weights, bones = variable_rows(blend.path, positions_size // POSITION_STRIDE, ids)
        else:
            stride = 4 if count == 1 else count * 8
            if blend.path.stat().st_size != positions_size // POSITION_STRIDE * stride:
                raise DumpError("Unexpected cs-t1 blend buffer size")
            bones = rows(blend.path, stride, 0 if count == 1 else count * 4, "<u4", count, ids)
            weights = np.ones(bones.shape, dtype=np.float32) if count == 1 else rows(
                blend.path, stride, 0, "<f4", count, ids
            )
        active = weights > 0
        if (
            not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1.00001)
            or not np.allclose(weights.sum(axis=1), 1, atol=0.001)
            or (palette and np.any(bones[active] >= palette.path.stat().st_size // 64))
        ):
            raise DumpError("Invalid APMI blend weights or bone indices")
        return weights, bones

    def component_key(self, draw, skinning):
        slots = self.resources[draw.call]
        return (
            skinning.call, slots["vb0", "buf"].path, slots["ib", "buf"].path,
            draw.first, draw.count, draw.base,
        )

    def parts(self, pre_skinning, import_weights, include_without_uv, draw_filter, import_textures):
        calls, hashes = parse_filter(draw_filter)
        groups, errors, reported = {}, [], set()
        for draw, shader in self.unsupported:
            ib = self.resources[draw.call]["ib", "buf"].hash
            key = ib, draw.first, draw.count, shader
            if ((calls or hashes) and draw.call not in calls and ib not in hashes) or key in reported:
                continue
            reported.add(key)
            errors.append(f"{draw.call:06d}: Unsupported skinning shader {shader} for IB {ib}")
        for draw, skinning in self.draws:
            slots = self.resources[draw.call]
            if (calls or hashes) and draw.call not in calls and slots["ib", "buf"].hash not in hashes:
                continue
            key = self.component_key(draw, skinning)
            groups.setdefault(key, []).append((draw, skinning))

        parts = []
        for group in groups.values():
            candidates = []
            for draw, skinning in group:
                slots = self.resources[draw.call]
                try:
                    part = self.decode_part(draw, skinning, pre_skinning, import_weights)
                except (OSError, ValueError, KeyError) as error:
                    errors.append(f"{draw.call:06d}: {error}")
                    continue
                if 0 not in part.uvs and not (include_without_uv or calls or hashes):
                    continue
                if import_textures:
                    for (slot, ext), resource in slots.items():
                        if slot.startswith("ps-t") and ext == "dds":
                            part.textures[int(slot[4:])] = resource
                candidates.append(part)
            if candidates:
                parts.append(max(candidates, key=lambda part: (
                    0 in part.uvs, len(part.uvs),
                    len(part.textures), part.color is not None,
                )))
        return sorted(parts, key=lambda part: part.draw.call), errors
