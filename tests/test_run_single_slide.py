import unittest
from unittest.mock import MagicMock, patch

import run_single_slide as single_mod


class _SlideContext:
    def __init__(self, slide):
        self.slide = slide

    def __enter__(self):
        return self.slide

    def __exit__(self, exc_type, exc, tb):
        return False


class _SegModel:
    def __init__(self, target_mag=1.25):
        self.target_mag = target_mag


class TestRunSingleSlide(unittest.TestCase):
    def _base_args(self, segmenter: str):
        class Args:
            pass

        args = Args()
        args.gpu = 1
        args.slide_path = "/tmp/fake.svs"
        args.job_dir = "/tmp/job"
        args.segmentation_source = "model"
        args.patch_encoder = "uni_v1"
        args.mag = 20
        args.patch_size = 256
        args.segmenter = segmenter
        args.seg_conf_thresh = 0.5
        args.remove_holes = False
        args.remove_artifacts = False
        args.remove_penmarks = False
        args.custom_mpp_keys = None
        args.overlap = 0
        args.batch_size = 4
        args.validation_mode = False
        args.annotation_vote_paths = None
        args.manual_tissue_mask_path = None
        args.manual_mask_source_level = 4
        args.manual_mask_target_level = 0
        args.manual_mask_tissue_thr = 0
        args.min_high_confidence_proportion = 0.5
        args.max_low_confidence_proportion = 0.1
        return args

    def test_process_slide_uses_cpu_for_otsu(self):
        args = self._base_args("otsu")
        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.segmentation_model_factory", return_value=_SegModel(target_mag=1.25)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            single_mod.process_slide(args)

        _, seg_kwargs = slide.segment_tissue.call_args
        self.assertEqual(seg_kwargs["device"], "cpu")
        slide.segment_tissue_from_manual_mask.assert_not_called()

    def test_process_slide_uses_gpu_for_hest(self):
        args = self._base_args("hest")
        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.segmentation_model_factory", return_value=_SegModel(target_mag=10)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            single_mod.process_slide(args)

        _, seg_kwargs = slide.segment_tissue.call_args
        self.assertEqual(seg_kwargs["device"], "cuda:1")
        slide.segment_tissue_from_manual_mask.assert_not_called()

    def test_process_slide_manual_mask_requires_explicit_path(self):
        args = self._base_args("hest")
        args.segmentation_source = "manual_mask"
        args.manual_tissue_mask_path = None

        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            with self.assertRaises(ValueError):
                single_mod.process_slide(args)

    def test_process_slide_manual_mask_dispatches_with_expected_args(self):
        args = self._base_args("hest")
        args.segmentation_source = "manual_mask"
        args.manual_tissue_mask_path = "/tmp/manual_mask.zarr"
        args.manual_mask_source_level = 5
        args.manual_mask_target_level = 0
        args.manual_mask_tissue_thr = 2

        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            single_mod.process_slide(args)

        slide.segment_tissue.assert_not_called()
        _, manual_kwargs = slide.segment_tissue_from_manual_mask.call_args
        self.assertEqual(manual_kwargs["mask_path"], "/tmp/manual_mask.zarr")
        self.assertEqual(manual_kwargs["source_level"], 5)
        self.assertEqual(manual_kwargs["target_level"], 0)
        self.assertEqual(manual_kwargs["tissue_thr"], 2)
        self.assertTrue(manual_kwargs["holes_are_tissue"])

    def test_process_slide_passes_validation_filter_args(self):
        args = self._base_args("hest")
        args.validation_mode = True
        args.annotation_vote_paths = "/tmp/a_votes.tif;/tmp/b_votes.tif"
        args.min_high_confidence_proportion = 0.6
        args.max_low_confidence_proportion = 0.05

        slide = MagicMock()
        slide.name = "fake"
        slide.extract_tissue_coords.return_value = "/tmp/job/coords.h5"
        slide.visualize_coords.return_value = "/tmp/job/viz.jpg"

        with patch("run_single_slide.load_wsi", return_value=_SlideContext(slide)), \
             patch("run_single_slide.segmentation_model_factory", return_value=_SegModel(target_mag=10)), \
             patch("run_single_slide.encoder_factory", return_value=MagicMock()):
            single_mod.process_slide(args)

        _, coords_kwargs = slide.extract_tissue_coords.call_args
        self.assertTrue(coords_kwargs["is_validation"])
        self.assertEqual(coords_kwargs["annotation_vote_paths"], args.annotation_vote_paths)
        self.assertEqual(coords_kwargs["min_high_confidence_proportion"], 0.6)
        self.assertEqual(coords_kwargs["max_low_confidence_proportion"], 0.05)


if __name__ == "__main__":
    unittest.main()
