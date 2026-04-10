import os
import tempfile
import unittest

import geopandas as gpd
import h5py
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon

from trident.IO import read_coords
from trident.wsi_objects.ImageWSI import ImageWSI


class DummyPatchEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_name = "dummy_patch"
        self.precision = torch.float32
        self.embedding_dim = 4

    @staticmethod
    def eval_transforms(img):
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def forward(self, x):
        # Deterministic lightweight embedding with shape [B, 4]
        bsz = x.shape[0]
        pooled = x.mean(dim=(2, 3))
        out = torch.zeros((bsz, self.embedding_dim), dtype=pooled.dtype, device=pooled.device)
        out[:, : min(pooled.shape[1], self.embedding_dim)] = pooled[:, : min(pooled.shape[1], self.embedding_dim)]
        return out


class DummySlideEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.precision = torch.float32
        self.called = False

    def forward(self, batch, device="cpu"):
        self.called = True
        # Return one deterministic slide embedding
        return torch.zeros((1, 8), dtype=torch.float32, device=device)


class TestEmptyCoordsPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir_ctx = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_ctx.name
        self.slide_path = os.path.join(self.tmpdir, "synthetic_slide.png")

        # Synthetic 20x-equivalent image with sparse signal.
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        img[100:120, 100:120, :] = 0
        Image.fromarray(img).save(self.slide_path)

    def tearDown(self):
        self.tmpdir_ctx.cleanup()

    @staticmethod
    def _tiny_tissue_mask():
        return gpd.GeoDataFrame(
            geometry=[
                Polygon(
                    [
                        (100, 100),
                        (120, 100),
                        (120, 120),
                        (100, 120),
                        (100, 100),
                    ]
                )
            ]
        )

    def _build_wsi_with_mask(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        wsi.gdf_contours = self._tiny_tissue_mask()
        return wsi

    def _write_vote_map(self, array: np.ndarray, name: str) -> str:
        vote_path = os.path.join(self.tmpdir, name)
        Image.fromarray(array.astype(np.uint8), mode="L").save(vote_path)
        return vote_path

    def test_high_min_tissue_proportion_produces_empty_coords(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            overlap=0,
            min_tissue_proportion=0.25,
        )
        attrs, coords = read_coords(coords_path)
        self.assertEqual(coords.shape, (0, 2))
        self.assertEqual(attrs["patch_size"], 256)

    def test_visualize_coords_handles_empty_coords(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            min_tissue_proportion=0.25,
        )
        viz_dir = os.path.join(self.tmpdir, "viz")
        viz_path = wsi.visualize_coords(coords_path=coords_path, save_patch_viz=viz_dir)
        self.assertTrue(os.path.exists(viz_path))
        with Image.open(viz_path) as img:
            self.assertGreater(img.size[0], 0)
            self.assertGreater(img.size[1], 0)

    def test_extract_patch_features_handles_empty_coords(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            min_tissue_proportion=0.25,
        )

        feats_dir = os.path.join(self.tmpdir, "features")
        out_path = wsi.extract_patch_features(
            patch_encoder=DummyPatchEncoder(),
            coords_path=coords_path,
            save_features=feats_dir,
            device="cpu",
            saveas="h5",
            batch_limit=16,
        )
        self.assertTrue(os.path.exists(out_path))
        with h5py.File(out_path, "r") as f:
            self.assertEqual(f["coords"].shape, (0, 2))
            self.assertEqual(f["features"].shape, (0, 4))

    def test_extract_slide_features_skips_encoder_on_empty_patch_features(self):
        wsi = self._build_wsi_with_mask()
        coords_path = wsi.extract_tissue_coords(
            target_mag=1.25,
            patch_size=256,
            save_coords=self.tmpdir,
            min_tissue_proportion=0.25,
        )
        patch_feats_path = wsi.extract_patch_features(
            patch_encoder=DummyPatchEncoder(),
            coords_path=coords_path,
            save_features=os.path.join(self.tmpdir, "features"),
            device="cpu",
            saveas="h5",
            batch_limit=16,
        )

        slide_encoder = DummySlideEncoder()
        slide_out = wsi.extract_slide_features(
            patch_features_path=patch_feats_path,
            slide_encoder=slide_encoder,
            save_features=os.path.join(self.tmpdir, "slide_features"),
            device="cpu",
        )
        self.assertTrue(os.path.exists(slide_out))
        self.assertFalse(slide_encoder.called)
        with h5py.File(slide_out, "r") as f:
            self.assertEqual(f["features"].shape, (0,))
            self.assertEqual(f["coords"].shape, (0, 2))

    def test_validation_confidence_filter_keeps_only_high_carcinoma_agreement_patches(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)

        vote_map = np.zeros((1024, 1024), dtype=np.uint8)
        vote_map[:512, :512] = 3
        vote_map[:512, 512:] = 24
        vote_map[512:, :512] = 21
        vote_map[512:712, 512:] = 24
        vote_path = self._write_vote_map(vote_map, "votes.tif")

        coords_path = wsi.extract_tissue_coords(
            target_mag=20,
            patch_size=512,
            save_coords=self.tmpdir,
            max_white_proportion=1.0,
            is_validation=True,
            annotation_vote_paths=vote_path,
            min_high_confidence_proportion=0.5,
            max_low_confidence_proportion=0.1,
        )

        attrs, coords = read_coords(coords_path)
        np.testing.assert_array_equal(
            coords,
            np.array([[0, 0], [512, 0], [512, 512]], dtype=np.int64),
        )
        self.assertEqual(attrs["annotation_vote_max_count"], 3)
        self.assertEqual(attrs["annotation_prefilter_patch_count"], 4)
        self.assertEqual(attrs["annotation_postfilter_patch_count"], 3)
        self.assertEqual(
            attrs["annotation_vote_interpretation"],
            "compact_soft_label_carcinoma_votes",
        )
        self.assertTrue(attrs["annotation_background_is_high_confidence"])

    def test_validation_confidence_filter_drops_background_only_slides(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        vote_path = self._write_vote_map(
            np.zeros((1024, 1024), dtype=np.uint8),
            "background_only_votes.tif",
        )

        coords_path = wsi.extract_tissue_coords(
            target_mag=20,
            patch_size=512,
            save_coords=self.tmpdir,
            max_white_proportion=1.0,
            is_validation=True,
            annotation_vote_paths=vote_path,
            min_high_confidence_proportion=0.5,
            max_low_confidence_proportion=0.1,
        )

        attrs, coords = read_coords(coords_path)
        self.assertEqual(coords.shape, (0, 2))
        self.assertEqual(attrs["annotation_vote_max_count"], 0)
        self.assertEqual(attrs["annotation_postfilter_patch_count"], 0)
        self.assertFalse(attrs["annotation_background_is_high_confidence"])

    def test_reviewed_benign_background_only_slide_saves_confident_background_stats(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)

        coords_path = wsi.extract_tissue_coords(
            target_mag=20,
            patch_size=512,
            save_coords=self.tmpdir,
            max_white_proportion=1.0,
            annotation_background_only=True,
        )

        attrs, coords = read_coords(coords_path)
        np.testing.assert_array_equal(
            coords,
            np.array([[0, 0], [512, 0], [0, 512], [512, 512]], dtype=np.int64),
        )
        self.assertTrue(attrs["annotation_statistics_available"])
        self.assertTrue(attrs["annotation_background_is_high_confidence"])
        self.assertTrue(attrs["annotation_background_only_slide"])
        self.assertEqual(attrs["annotation_vote_max_count"], 0)
        self.assertEqual(attrs["annotation_vote_interpretation"], "reviewed_benign_background_only")

        with h5py.File(coords_path, "r") as f:
            raw_hist = f["label_hist_raw_compact"][:]
            effective_hist = f["label_hist_effective_carcinoma"][:]
            confident_pixels = f["label_confident_pixel_count"][:]
            low_confidence_pixels = f["label_low_confidence_pixel_count"][:]
            patch_area_pixels = f["patch_area_pixels"][:]

            np.testing.assert_array_equal(raw_hist[:, 0], np.full((4,), 262144, dtype=np.uint32))
            np.testing.assert_array_equal(raw_hist[:, 1:], np.zeros((4, 15), dtype=np.uint32))
            np.testing.assert_array_equal(
                effective_hist,
                np.tile(np.array([[262144, 0, 0, 0]], dtype=np.uint32), (4, 1)),
            )
            np.testing.assert_array_equal(confident_pixels, np.full((4,), 262144, dtype=np.uint32))
            np.testing.assert_array_equal(low_confidence_pixels, np.zeros((4,), dtype=np.uint32))
            np.testing.assert_array_equal(patch_area_pixels, np.full((4,), 262144, dtype=np.uint32))

    def test_validation_confidence_filter_keeps_reviewed_benign_background_only_slides(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)

        coords_path = wsi.extract_tissue_coords(
            target_mag=20,
            patch_size=512,
            save_coords=self.tmpdir,
            max_white_proportion=1.0,
            is_validation=True,
            annotation_background_only=True,
            min_high_confidence_proportion=0.5,
            max_low_confidence_proportion=0.1,
        )

        attrs, coords = read_coords(coords_path)
        np.testing.assert_array_equal(
            coords,
            np.array([[0, 0], [512, 0], [0, 512], [512, 512]], dtype=np.int64),
        )
        self.assertTrue(attrs["annotation_background_is_high_confidence"])
        self.assertTrue(attrs["annotation_background_only_slide"])
        self.assertEqual(attrs["annotation_vote_max_count"], 0)
        self.assertEqual(attrs["annotation_prefilter_patch_count"], 4)
        self.assertEqual(attrs["annotation_postfilter_patch_count"], 4)

    def test_patch_stats_and_white_filter_are_saved_and_mirrored_to_feature_h5(self):
        img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
        img[:300, :300, :] = 64
        Image.fromarray(img).save(self.slide_path)

        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)

        vote_map = np.zeros((1024, 1024), dtype=np.uint8)
        vote_map[:200, :200] = 3
        vote_map[:200, 200:400] = 13
        vote_map[200:400, :200] = 21
        vote_path = self._write_vote_map(vote_map, "stats_votes.tif")

        coords_path = wsi.extract_tissue_coords(
            target_mag=20,
            patch_size=512,
            save_coords=self.tmpdir,
            annotation_vote_paths=vote_path,
        )

        attrs, coords = read_coords(coords_path)
        np.testing.assert_array_equal(coords, np.array([[0, 0]], dtype=np.int64))
        self.assertTrue(attrs["annotation_statistics_available"])
        self.assertEqual(attrs["white_prefilter_patch_count"], 4)
        self.assertEqual(attrs["white_postfilter_patch_count"], 1)
        self.assertEqual(attrs["annotation_vote_max_count"], 3)
        self.assertTrue(attrs["annotation_background_is_high_confidence"])

        with h5py.File(coords_path, "r") as f:
            self.assertIn("label_hist_raw_compact", f)
            self.assertIn("label_hist_effective_carcinoma", f)
            self.assertIn("label_confident_pixel_count", f)
            self.assertIn("label_low_confidence_pixel_count", f)
            self.assertIn("patch_area_pixels", f)
            self.assertIn("white_pixel_fraction", f)

            raw_hist = f["label_hist_raw_compact"][:]
            effective_hist = f["label_hist_effective_carcinoma"][:]
            confident_pixels = f["label_confident_pixel_count"][:]
            low_confidence_pixels = f["label_low_confidence_pixel_count"][:]
            patch_area_pixels = f["patch_area_pixels"][:]
            white_fraction = f["white_pixel_fraction"][:]

            np.testing.assert_array_equal(
                f["label_hist_raw_compact"].attrs["label_values"],
                np.array([0, 1, 2, 3, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 29], dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                raw_hist[:, [0, 3, 6, 7]],
                np.array([[142144, 40000, 40000, 40000]], dtype=np.uint32),
            )
            np.testing.assert_array_equal(
                effective_hist,
                np.array([[142144, 0, 40000, 80000]], dtype=np.uint32),
            )
            np.testing.assert_array_equal(confident_pixels, np.array([222144], dtype=np.uint32))
            np.testing.assert_array_equal(low_confidence_pixels, np.array([40000], dtype=np.uint32))
            np.testing.assert_array_equal(patch_area_pixels, np.array([262144], dtype=np.uint32))
            self.assertLess(float(white_fraction[0]), 0.9)

        feats_dir = os.path.join(self.tmpdir, "features_with_stats")
        feats_path = wsi.extract_patch_features(
            patch_encoder=DummyPatchEncoder(),
            coords_path=coords_path,
            save_features=feats_dir,
            device="cpu",
            saveas="h5",
            batch_limit=16,
        )
        with h5py.File(feats_path, "r") as f:
            self.assertIn("label_hist_raw_compact", f)
            self.assertIn("white_pixel_fraction", f)
            np.testing.assert_array_equal(f["coords"][:], np.array([[0, 0]], dtype=np.int64))
            np.testing.assert_array_equal(
                f["label_hist_effective_carcinoma"][:],
                np.array([[142144, 0, 40000, 80000]], dtype=np.uint32),
            )

    def test_training_mode_ignores_annotation_confidence_filter(self):
        wsi = ImageWSI(slide_path=self.slide_path, mpp=0.5, lazy_init=False)
        vote_map = np.zeros((1024, 1024), dtype=np.uint8)
        vote_map[:512, :512] = 2
        vote_path = self._write_vote_map(vote_map, "train_votes.tif")

        coords_path = wsi.extract_tissue_coords(
            target_mag=20,
            patch_size=512,
            save_coords=self.tmpdir,
            max_white_proportion=1.0,
            is_validation=False,
            annotation_vote_paths=vote_path,
        )

        _, coords = read_coords(coords_path)
        self.assertEqual(coords.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
