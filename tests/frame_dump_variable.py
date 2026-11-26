import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np


sys.dont_write_bytecode = True
package = ModuleType("apmi_tools")
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package

from apmi_tools.frame_dump import VARIABLE_INFLUENCES, DumpError, FrameDump, Resource, Skinning


def packed(bone, weight):
    return (weight << 16) | bone


class VariableBlend(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.vertices = [
            [(7, 65535)],
            [(1, 13107), (2, 13107), (3, 13107), (4, 13107), (5, 13107)],
            [(bone, 8191) for bone in range(8)],
        ]
        self.vertices[2][0] = (0, 8198)

    def tearDown(self):
        self.temp.cleanup()

    def skinning(self, offsets=None):
        data = [packed(bone, weight) for vertex in self.vertices for bone, weight in vertex]
        if offsets is None:
            offsets = np.cumsum([len(self.vertices) + 1] + [len(vertex) for vertex in self.vertices])
        files = {
            "rest": np.zeros(len(self.vertices) * 10, dtype="<f4").tobytes(),
            "blend": np.asarray([*offsets, *data], dtype="<u4").tobytes(),
            "palette": np.zeros(8 * 16, dtype="<f4").tobytes(),
        }
        resources = []
        for name, content in files.items():
            path = self.root / f"{name}.buf"
            path.write_bytes(content)
            resources.append(Resource("00000000", path, ""))
        return Skinning(0, VARIABLE_INFLUENCES, *resources)

    def test_decodes_one_to_eight_influences(self):
        weights, bones = FrameDump.decode_weights(self.skinning(), np.array([0, 1, 2]))
        self.assertEqual(bones.shape, (3, 8))
        np.testing.assert_array_equal(bones[0], [7, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(bones[1, :5], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(bones[2], np.arange(8))
        np.testing.assert_array_equal((weights > 0).sum(axis=1), [1, 5, 8])
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)

    def test_decodes_only_requested_vertices(self):
        weights, bones = FrameDump.decode_weights(self.skinning(), np.array([2]))
        np.testing.assert_array_equal(bones[0], np.arange(8))

    def test_rejects_corrupt_offset_tables(self):
        valid = np.cumsum([4, 1, 5, 8])
        for offsets in (valid - 1, valid[[0, 2, 1, 3]], [4, 5, 5, 18]):
            with self.subTest(offsets=list(offsets)), self.assertRaises(DumpError):
                FrameDump.decode_weights(self.skinning(offsets), np.array([0, 1, 2]))


class UnsupportedSkinning(unittest.TestCase):
    def test_draw_from_unknown_skinning_shader_is_reported(self):
        shader = "0123456789abcdef"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "log.txt").write_text("\n".join([
                "000001 CSSetUnorderedAccessViews(StartSlot:0, NumUAVs:1, ppUnorderedAccessViews:0x0)",
                "       0: resource=0x0000000000001000",
                "000001 Dispatch(ThreadGroupCountX:1, ThreadGroupCountY:1, ThreadGroupCountZ:1)",
                "000002 IASetVertexBuffers(StartSlot:0, NumBuffers:1, ppVertexBuffers:0x0)",
                "       0: resource=0x0000000000001000",
                "000002 DrawIndexed(IndexCount:3, StartIndexLocation:0, BaseVertexLocation:0)",
            ]) + "\n", encoding="utf-8")
            for name in (
                f"000001-u0=11111111-cs={shader}.buf", f"000001-cs-t2=22222222-cs={shader}.buf",
                "000002-vb0=11111111-vs=0000000000000000.buf", "000002-ib=33333333-vs=0000000000000000.buf",
            ):
                (root / name).write_bytes(b"\0" * 64)
            parts, errors = FrameDump(root / "log.txt").parts(False, True, False, "", False)
        self.assertEqual(parts, [])
        self.assertEqual(errors, [f"000002: Unsupported skinning shader {shader} for IB 33333333"])


if __name__ == "__main__":
    unittest.main()
