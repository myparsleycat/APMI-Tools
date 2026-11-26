import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


sys.dont_write_bytecode = True
package = ModuleType("apmi_tools")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package

from apmi_tools.mod_export import MeshBuffers
from apmi_tools.skinning_variants import BLEND, BLEND8, SkinProfile, match_bones, merged_variant, remap_blend


def part(bones):
    return SimpleNamespace(
        vertex_ids=np.array([0, 9, 12]),
        triangles=np.array([[0, 1, 2]]),
        bones=np.asarray(bones, dtype="<u4"),
        weights=np.array([[.25, .75], [1, 0], [1, 0]], dtype="<f4"),
    )


class FieldVariants(unittest.TestCase):
    def test_equal_palette_sizes_can_have_different_slots(self):
        source = part([[0, 1], [0, 0], [1, 0]])
        field = part([[1, 0], [1, 0], [0, 0]])
        blend = np.zeros(1, dtype=BLEND)
        blend["weights"][0, :2] = (.4, .6)
        blend["indices"][0, :2] = (1, 0)
        data, missing = remap_blend(blend.tobytes(), match_bones(source, field), BLEND)
        self.assertFalse(missing)
        remapped = np.frombuffer(data, dtype=BLEND)
        np.testing.assert_array_equal(remapped["indices"][0], [0, 1, 0, 0])
        np.testing.assert_array_equal(remapped["weights"], blend["weights"])

    def test_eight_influence_blend_remaps_upper_columns(self):
        blend = np.zeros(2, dtype=BLEND8)
        blend["weights"][0] = 1 / 8
        blend["indices"][0] = np.arange(8)
        blend["weights"][1, 0] = 1
        data, missing = remap_blend(blend.tobytes(), {bone: 10 + bone for bone in range(8)}, BLEND8)
        self.assertFalse(missing)
        remapped = np.frombuffer(data, dtype=BLEND8)
        np.testing.assert_array_equal(remapped["indices"][0], np.arange(10, 18))
        np.testing.assert_array_equal(remapped["indices"][1], [10, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(remapped["weights"], blend["weights"])

    def test_missing_bones_are_supplied_only_by_verified_donors(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "palette.buf"
            native = np.tile(np.eye(4, dtype="<f4"), (2, 1, 1))
            native[1, 3, 0] = 2
            file.write_bytes(native.tobytes())
            source = part([[0, 1], [0, 0], [1, 0]])
            source.index_hash = "12345678"
            source.draw = SimpleNamespace(first=0, count=3, base=0)
            source.skinning = SimpleNamespace(palette=SimpleNamespace(path=file), influence_count=4)
            field = SkinProfile("22222222", 1, 1)
            blend = np.zeros(1, dtype=BLEND)
            blend["weights"][0, :2] = (.5, .5)
            blend["indices"][0, :2] = (0, 1)
            mod = MeshBuffers(1, 3, {"Blend": blend.tobytes()})
            outcome, missing = merged_variant(mod, source, field, {0: 0}, [])
            self.assertIsNone(outcome)
            self.assertEqual(missing, [1])
            self.assertEqual(set(mod.files), {"Blend"})

            donor = (("33333333", 0, 6, 0), SkinProfile("44444444", 4, 8), native, {1: 7}, [0, 4])
            outcome, missing = merged_variant(mod, source, field, {0: 0}, [donor])
            self.assertFalse(missing)
            self.assertEqual(outcome.bone_count, 2)
            self.assertEqual(outcome.donors[1][0].component, donor[0])
            np.testing.assert_array_equal(np.frombuffer(mod.files["BoneMap0"], "<u4"), [0, 0])
            np.testing.assert_array_equal(np.frombuffer(mod.files["BoneMap1"], "<u4"), [7, 1])


if __name__ == "__main__":
    unittest.main()
