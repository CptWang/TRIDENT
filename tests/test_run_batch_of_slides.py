import unittest
from unittest.mock import patch

import run_batch_of_slides as batch_mod


class _DummySegmentationModel:
    def __init__(self, target_mag=1.25):
        self.target_mag = target_mag


class _DummyProcessor:
    def __init__(self):
        self.calls = []

    def run_segmentation_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def run_patching_job(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestRunBatchOfSlides(unittest.TestCase):
    def _base_args(self, segmenter: str):
        class Args:
            pass

        args = Args()
        args.task = "seg"
        args.segmentation_source = "model"
        args.segmenter = segmenter
        args.seg_conf_thresh = 0.5
        args.remove_holes = False
        args.remove_artifacts = False
        args.remove_penmarks = False
        args.seg_batch_size = None
        args.custom_list_of_wsis = None
        args.manual_tissue_mask_column = None
        args.manual_mask_source_level = 4
        args.manual_mask_target_level = 0
        args.manual_mask_tissue_thr = 0
        args.batch_size = 8
        args.gpu = 2
        args.mag = 20
        args.patch_size = 256
        args.overlap = 0
        args.coords_dir = None
        args.min_tissue_proportion = 0.0
        args.validation_mode = False
        args.min_high_confidence_proportion = 0.5
        args.max_low_confidence_proportion = 0.1
        return args

    def test_run_task_seg_uses_cpu_for_otsu(self):
        processor = _DummyProcessor()
        args = self._base_args("otsu")

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=_DummySegmentationModel()):
            batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["segmentation_source"], "model")

    def test_run_task_seg_uses_gpu_for_hest(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")

        with patch("trident.segmentation_models.load.segmentation_model_factory", return_value=_DummySegmentationModel(target_mag=10)):
            batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["device"], "cuda:2")
        self.assertEqual(kwargs["segmentation_source"], "model")

    def test_run_task_seg_manual_mask_requires_custom_list(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")
        args.segmentation_source = "manual_mask"
        args.manual_tissue_mask_column = "manual_mask"

        with self.assertRaises(ValueError):
            batch_mod.run_task(processor, args)

    def test_run_task_seg_manual_mask_requires_mask_column(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")
        args.segmentation_source = "manual_mask"
        args.custom_list_of_wsis = "/tmp/wsis.csv"
        args.manual_tissue_mask_column = None

        with self.assertRaises(ValueError):
            batch_mod.run_task(processor, args)

    def test_run_task_seg_manual_mask_dispatches_to_processor(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")
        args.segmentation_source = "manual_mask"
        args.custom_list_of_wsis = "/tmp/wsis.csv"
        args.manual_tissue_mask_column = "manual_mask"
        args.manual_mask_source_level = 5
        args.manual_mask_target_level = 0
        args.manual_mask_tissue_thr = 1

        batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertEqual(kwargs["segmentation_source"], "manual_mask")
        self.assertEqual(kwargs["manual_mask_source_level"], 5)
        self.assertEqual(kwargs["manual_mask_target_level"], 0)
        self.assertEqual(kwargs["manual_mask_tissue_thr"], 1)
        self.assertIsNone(kwargs["segmentation_model"])

    def test_run_task_coords_passes_validation_filter_args(self):
        processor = _DummyProcessor()
        args = self._base_args("hest")
        args.task = "coords"
        args.validation_mode = True
        args.min_high_confidence_proportion = 0.6
        args.max_low_confidence_proportion = 0.05

        batch_mod.run_task(processor, args)

        self.assertEqual(len(processor.calls), 1)
        kwargs = processor.calls[0][1]
        self.assertTrue(kwargs["validation_mode"])
        self.assertEqual(kwargs["min_high_confidence_proportion"], 0.6)
        self.assertEqual(kwargs["max_low_confidence_proportion"], 0.05)


if __name__ == "__main__":
    unittest.main()
