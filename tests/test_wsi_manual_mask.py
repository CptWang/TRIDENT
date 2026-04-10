import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from trident.wsi_objects.WSI import WSI

wsi_module = importlib.import_module("trident.wsi_objects.WSI")


class _FakeZarrArray:
    def __init__(self, data: np.ndarray):
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, item):
        return self._data[item]


class _FakeZarrGroup:
    def __init__(self, arrays: dict[str, _FakeZarrArray]):
        self._arrays = arrays

    def array_keys(self):
        return list(self._arrays.keys())

    def __getitem__(self, key: str):
        return self._arrays[key]


def _fake_zarr_module(root_group: _FakeZarrGroup) -> types.SimpleNamespace:
    fake_module = types.SimpleNamespace()

    class _ArrayType:
        pass

    fake_module.Array = _ArrayType
    fake_module.open = lambda *_args, **_kwargs: root_group
    return fake_module


class _DummyManualMaskWSI(WSI):
    def _lazy_initialize(self) -> None:
        if self._initialized:
            return
        super()._lazy_initialize()
        self.width = 4
        self.height = 4
        self.mpp = 0.5
        self._initialized = True

    def get_thumbnail(self, size):
        return Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))

    def get_dimensions(self):
        return self.width, self.height

    def get_best_level_and_custom_downsample(self, downsample, tolerance=0.1):
        return 0, 1.0

    def read_region(self, location, level, size, read_as="numpy"):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def release(self):
        return None


class TestManualMaskReconstruction(unittest.TestCase):
    def test_reconstruct_manual_mask_falls_back_to_nearest_available_source_level(self):
        source = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        target = np.zeros((4, 4), dtype=np.uint8)
        root = _FakeZarrGroup({"0": _FakeZarrArray(source)})

        fake_zarr = _fake_zarr_module(root)
        with patch.dict(sys.modules, {"zarr": fake_zarr}):
            with self.assertWarnsRegex(UserWarning, r"source_level=4.*Falling back to available level 0"):
                mask = WSI._reconstruct_manual_mask_2d_from_zarr(
                    mask_path="/tmp/unused.zarr",
                    source_level=4,
                    target_level=0,
                    tissue_thr=0,
                )

        expected = np.array([[False, True], [True, False]])
        self.assertEqual(mask.shape, (2, 2))
        self.assertTrue(np.array_equal(mask, expected))

    def test_reconstruct_manual_mask_from_2d_source_resizes_nearest(self):
        source = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        target = np.zeros((4, 4), dtype=np.uint8)
        root = _FakeZarrGroup({"4": _FakeZarrArray(source), "0": _FakeZarrArray(target)})

        fake_zarr = _fake_zarr_module(root)
        with patch.dict(sys.modules, {"zarr": fake_zarr}):
            mask = WSI._reconstruct_manual_mask_2d_from_zarr(
                mask_path="/tmp/unused.zarr",
                source_level=4,
                target_level=0,
                tissue_thr=0,
            )

        expected = np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [True, True, False, False],
                [True, True, False, False],
            ]
        )
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(np.array_equal(mask, expected))

    def test_reconstruct_manual_mask_from_3d_source_projects_and_resizes(self):
        source = np.array(
            [
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=np.uint8,
        )
        target = np.zeros((2, 4, 4), dtype=np.uint8)
        root = _FakeZarrGroup({"4": _FakeZarrArray(source), "0": _FakeZarrArray(target)})

        fake_zarr = _fake_zarr_module(root)
        with patch.dict(sys.modules, {"zarr": fake_zarr}):
            mask = WSI._reconstruct_manual_mask_2d_from_zarr(
                mask_path="/tmp/unused.zarr",
                source_level=4,
                target_level=0,
                tissue_thr=0,
            )

        expected = np.array(
            [
                [False, False, True, True],
                [False, False, True, True],
                [True, True, False, False],
                [True, True, False, False],
            ]
        )
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(np.array_equal(mask, expected))


class TestManualMaskSegmentationAPI(unittest.TestCase):
    def test_segment_tissue_from_manual_mask_shape_mismatch_raises(self):
        wsi = _DummyManualMaskWSI(slide_path="dummy.svs", lazy_init=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "manual.zarr")
            with open(mask_path, "w", encoding="utf-8") as handle:
                handle.write("")

            with patch.object(
                _DummyManualMaskWSI,
                "_reconstruct_manual_mask_2d_from_zarr",
                return_value=np.zeros((3, 3), dtype=bool),
            ):
                with self.assertRaises(ValueError):
                    wsi.segment_tissue_from_manual_mask(mask_path=mask_path, job_dir=None)

    def test_segment_tissue_from_manual_mask_returns_gdf_in_memory(self):
        wsi = _DummyManualMaskWSI(slide_path="dummy.svs", lazy_init=True)
        in_memory_gdf = MagicMock(name="gdf")

        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "manual.zarr")
            with open(mask_path, "w", encoding="utf-8") as handle:
                handle.write("")

            with patch.object(
                _DummyManualMaskWSI,
                "_reconstruct_manual_mask_2d_from_zarr",
                return_value=np.zeros((4, 4), dtype=bool),
            ), patch.object(wsi_module, "mask_to_gdf", return_value=in_memory_gdf):
                out = wsi.segment_tissue_from_manual_mask(mask_path=mask_path, job_dir=None)

        self.assertIs(out, in_memory_gdf)
        self.assertIs(wsi.gdf_contours, in_memory_gdf)
        self.assertIsNone(wsi.tissue_seg_path)


if __name__ == "__main__":
    unittest.main()
