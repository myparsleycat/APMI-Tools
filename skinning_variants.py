from dataclasses import dataclass

import numpy as np

from .frame_dump import DumpError


BLEND = np.dtype([("weights", "<f4", 4), ("indices", "<u4", 4)])
WEIGHT_TOLERANCE = 2e-5
POSE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class SkinProfile:
    blend_hash: str
    influences: int
    bones: int

    @classmethod
    def from_skinning(cls, skinning):
        if not skinning.blend or not skinning.palette:
            raise DumpError("Missing native blend buffer or bone palette")
        blend_hash = skinning.blend.hash
        if len(blend_hash) != 8 or any(digit not in "0123456789abcdef" for digit in blend_hash):
            raise DumpError("The native blend buffer has no usable hash")
        size = skinning.palette.path.stat().st_size
        if not size or size % 64 or size // 64 >= 2**21:
            raise DumpError("Invalid native bone palette size")
        return cls(blend_hash, skinning.influence_count, size // 64)


@dataclass(frozen=True)
class BoneSource:
    component: tuple[str, int, int, int]
    profile: SkinProfile
    slot: int


@dataclass(frozen=True)
class Variant:
    blend: str
    donors: tuple[tuple[BoneSource, str], ...] = ()
    bone_count: int = 0


def draw_key(ib_hash, draw):
    return ib_hash, draw.first, draw.count, draw.base


def weight_columns(part):
    active = part.weights > 0
    bones = np.unique(part.bones[active])
    weights = np.zeros((len(bones), len(part.vertex_ids)), dtype=np.float32)
    vertices, slots = np.nonzero(active)
    rows = np.searchsorted(bones, part.bones[vertices, slots])
    normalized = part.weights / part.weights.sum(axis=1, keepdims=True)
    np.add.at(weights, (rows, vertices), normalized[vertices, slots])
    return bones, weights


def match_bones(reference, field):
    if not np.array_equal(reference.vertex_ids, field.vertex_ids) or not np.array_equal(
        reference.triangles, field.triangles,
    ):
        return {}

    source_ids, source_weights = weight_columns(reference)
    target_ids, target_weights = weight_columns(field)
    source_counts = np.count_nonzero(source_weights, axis=1)
    target_counts = np.count_nonzero(target_weights, axis=1)
    candidates = {}
    claims = {}
    for bone, weights, count in zip(source_ids, source_weights, source_counts):
        matches = []
        for row in np.flatnonzero(target_counts == count):
            target = target_weights[row]
            if np.array_equal(weights > 0, target > 0) and np.allclose(
                weights, target, rtol=0, atol=WEIGHT_TOLERANCE,
            ):
                slot = int(target_ids[row])
                matches.append(slot)
                claims[slot] = claims.get(slot, 0) + 1
        if len(matches) == 1:
            candidates[int(bone)] = matches[0]
    return {bone: slot for bone, slot in candidates.items() if claims[slot] == 1}


def palette(part):
    return np.fromfile(part.skinning.palette.path, dtype="<f4").reshape(-1, 4, 4)


def verify_field_pose(reference, field, posed, mapping):
    used = reference.weights > 0
    if any(int(bone) not in mapping for bone in np.unique(reference.bones[used])):
        return False
    remap = np.zeros(len(palette(reference)), dtype=np.int32)
    for bone, slot in mapping.items():
        remap[bone] = slot
    slots = remap[np.where(used, reference.bones, 0)]
    weights = reference.weights / reference.weights.sum(axis=1, keepdims=True)
    matrices = np.sum(palette(field)[slots] * weights[:, :, None, None], axis=1)
    xyz = np.einsum("ni,nij->nj", field.positions, matrices[:, :3, :3]) + matrices[:, 3, :3]
    return np.isfinite(xyz).all() and np.max(np.linalg.norm(xyz - posed.positions, axis=1)) <= POSE_TOLERANCE


def field_draws(dump):
    candidates = {}
    ambiguous = set()
    for draw, skin in dump.draws:
        ib = dump.resources[draw.call]["ib", "buf"]
        key = draw_key(ib.hash, draw)
        old = candidates.setdefault(key, (draw, skin))
        if dump.component_key(*old) != dump.component_key(draw, skin):
            ambiguous.add(key)
    return {key: value for key, value in candidates.items() if key not in ambiguous}


def original_pair(reference_dump, field_dump, source, target):
    reference = reference_dump.decode_part(*source, True, True)
    field = field_dump.decode_part(*target, True, True)
    reference_profile = SkinProfile.from_skinning(reference.skinning)
    field_profile = SkinProfile.from_skinning(field.skinning)
    if not np.array_equal(reference.vertex_ids, field.vertex_ids) or not np.array_equal(
        reference.triangles, field.triangles,
    ):
        raise DumpError("The field draw has different source topology")
    mapping = match_bones(reference, field)
    posed = field_dump.decode_part(*target, False, False)
    verified = verify_field_pose(reference, field, posed, mapping)
    return reference, reference_profile, field_profile, mapping if verified else {}


def remap_blend(data, mapping):
    # Adapted from EFMI-Tools build_lod_buffers (eaa9036), GPL-3.0.
    blend = np.frombuffer(data, dtype=BLEND).copy()
    active = blend["weights"] > 0
    used = np.unique(blend["indices"][active])
    missing = [int(bone) for bone in used if int(bone) not in mapping]
    if missing:
        return None, missing
    indices = blend["indices"]
    remap = np.zeros(int(used[-1]) + 1, dtype="<u4")
    for bone in used:
        remap[bone] = mapping[int(bone)]
    indices[active] = remap[indices[active]]
    return blend.tobytes(), []


def actor_transform(dump, draw):
    resource = dump.resources[draw.call].get(("vs-cb1", "buf"))
    if not resource:
        return None
    with resource.path.open("rb") as stream:
        data = stream.read(128)
    if len(data) != 128:
        return None
    matrices = np.frombuffer(data, dtype="<f4").reshape(2, 4, 4)
    # AP captures carry an affine transform and its inverse in this VS constant buffer.
    if not np.isfinite(matrices).all() or not np.allclose(
        matrices[0] @ matrices[1], np.eye(4), rtol=0, atol=1e-3,
    ):
        return None
    return data


def donor_catalog(reference_dump, field_dump, targets, reference, target):
    reference_actor = actor_transform(reference_dump, reference.draw)
    field_actor = actor_transform(field_dump, target[0])
    if reference_actor is None or field_actor is None:
        return []

    donors = []
    reference_key = draw_key(reference.index_hash, reference.draw)
    for key, original in field_draws(reference_dump).items():
        field = targets.get(key)
        if field is None or key == reference_key:
            continue
        if actor_transform(reference_dump, original[0]) != reference_actor or actor_transform(
            field_dump, field[0],
        ) != field_actor:
            continue
        try:
            donor, _, profile, mapping = original_pair(reference_dump, field_dump, original, field)
        except (DumpError, KeyError, OSError):
            continue
        if not mapping:
            continue
        weights = donor.weights > 0
        counts = np.bincount(donor.bones[weights], minlength=len(palette(donor)))
        donors.append((key, profile, palette(donor), mapping, counts))
    return donors


def merged_variant(buffers, reference, profile, mapping, donors):
    # Adapted from EFMI-Tools build_merged_skeleton_vg_map (eaa9036), GPL-3.0.
    used_blend = np.frombuffer(buffers.files["Blend"], dtype=BLEND)
    used_bones = np.unique(used_blend["indices"][used_blend["weights"] > 0])
    reference_matrices = palette(reference)
    sources, missing = {}, []
    receiver = draw_key(reference.index_hash, reference.draw)
    for bone in used_bones:
        bone = int(bone)
        if bone in mapping:
            sources[bone] = BoneSource(receiver, profile, mapping[bone])
            continue
        candidates = []
        matrix = reference_matrices[bone]
        for key, donor_profile, matrices, slots, counts in donors:
            matches = np.flatnonzero(np.max(np.abs(matrices - matrix), axis=(1, 2)) <= 1e-6)
            if len(matches) == 1 and int(matches[0]) in slots and np.any(matrix):
                slot = int(matches[0])
                candidates.append((-int(counts[slot]), key, donor_profile, slots[slot]))
        if candidates:
            _, key, donor_profile, slot = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
            sources[bone] = BoneSource(key, donor_profile, slot)
        else:
            missing.append(bone)
    if missing:
        return None, missing

    compact = {bone: index for index, bone in enumerate(sources)}
    data, _ = remap_blend(buffers.files["Blend"], compact)
    buffers.files["BlendField"] = data
    groups = {}
    for bone, source in sources.items():
        groups.setdefault(BoneSource(source.component, source.profile, 0), []).append((source.slot, compact[bone]))
    tables = []
    for n, (source, pairs) in enumerate(groups.items()):
        name = f"BoneMap{n}"
        buffers.files[name] = np.asarray(pairs, dtype="<u4").tobytes()
        tables.append((source, name))
    return Variant("BlendField", tuple(tables), len(compact)), []


def add_variants(dumps, field_dump, targets, parts):
    from .mod_export import get_component

    for log, component, buffers in parts:
        source = component.draw, component.skinning
        profile = SkinProfile.from_skinning(component.skinning)
        buffers.variants = {profile: Variant("Blend")}
        target = targets.get(draw_key(component.ib, component.draw))
        if target is None:
            buffers.variant_warning = "No unique matching draw in the field capture"
            continue
        try:
            field_component = get_component(field_dump, target[0].call)
            if field_component.streams != component.streams:
                raise DumpError("The field draw uses a different vertex stream layout")
            reference, _, field_profile, mapping = original_pair(dumps[log], field_dump, source, target)
        except (DumpError, KeyError, OSError) as error:
            buffers.variant_warning = str(error)
            if target[1].blend and target[1].blend.hash == profile.blend_hash:
                buffers.variants.pop(profile)
            continue

        data, missing = remap_blend(buffers.files["Blend"], mapping)
        if field_profile == profile:
            if data != buffers.files["Blend"]:
                buffers.variants.pop(profile)
                buffers.variant_warning = "The source and field share an indistinguishable native skin profile"
            continue
        if missing:
            donors = donor_catalog(dumps[log], field_dump, targets, reference, target)
            if donors:
                merged, missing = merged_variant(buffers, reference, field_profile, mapping, donors)
            if missing:
                buffers.variant_warning = "Field palette has no verified mapping for bones " + ", ".join(map(str, missing))
                continue
            buffers.variants[field_profile] = merged
        else:
            buffers.files["BlendField"] = data
            buffers.variants[field_profile] = Variant("BlendField")
        component.passes.extend(field_component.passes)
        for digest, (paths, slots) in field_component.textures.items():
            source_paths, source_slots = component.textures.setdefault(digest, (set(), set()))
            source_paths.update(paths)
            source_slots.update(slots)
