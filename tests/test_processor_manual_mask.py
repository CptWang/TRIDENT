import importlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from trident import Processor

processor_module = importlib.import_module("trident.Processor")


class _Ctx:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyWSI:
    def __init__(self, name: str, ext: str, manual_tissue_mask_path: str):
        self.name = name
        self.ext = ext
        self.manual_tissue_mask_path = manual_tissue_mask_path
        self.manual_calls = []
        self.model_calls = []
        self.release_calls = 0

    def segment_tissue_from_manual_mask(self, **kwargs):
        self.manual_calls.append(kwargs)
        output_geojson = os.path.join(kwargs["job_dir"], "contours_geojson", f"{self.name}.geojson")
        os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
        with open(output_geojson, "w", encoding="utf-8") as handle:
            handle.write("{}")
        return output_geojson

    def segment_tissue(self, **kwargs):
        self.model_calls.append(kwargs)
        output_geojson = os.path.join(kwargs["job_dir"], "contours_geojson", f"{self.name}.geojson")
        os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
        with open(output_geojson, "w", encoding="utf-8") as handle:
            handle.write("{}")
        return output_geojson

    def release(self):
        self.release_calls += 1


class TestProcessorManualMask(unittest.TestCase):
    def test_init_raises_when_manual_mask_column_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "wsis.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("wsi\nslide.svs\n")

            with patch.object(
                processor_module,
                "collect_valid_slides",
                return_value=(["/tmp/slide.svs"], ["slide.svs"]),
            ), patch.object(
                processor_module,
                "load_wsi",
                return_value=_Ctx(MagicMock(name="slide")),
            ), patch.object(processor_module.os.path, "exists", return_value=False):
                with self.assertRaises(ValueError):
                    Processor(
                        job_dir=tmpdir,
                        wsi_source="/tmp/wsi",
                        wsi_ext=[".svs"],
                        custom_list_of_wsis=csv_path,
                        manual_tissue_mask_column="manual_mask",
                    )

    def test_init_marks_empty_label_rows_as_background_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "wsis.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "wsi,mpp,original_2d_nodo_label\n"
                    "/tmp/slide.svs,0.5,empty\n"
                )

            slide = MagicMock(name="slide")
            with patch.object(
                processor_module,
                "collect_valid_slides",
                return_value=(["/tmp/slide.svs"], ["slide.svs"]),
            ), patch.object(
                processor_module,
                "load_wsi",
                return_value=_Ctx(slide),
            ), patch.object(processor_module.os.path, "exists", return_value=False):
                Processor(
                    job_dir=tmpdir,
                    wsi_source="/tmp/wsi",
                    wsi_ext=[".svs"],
                    custom_list_of_wsis=csv_path,
                    annotation_vote_column="original_2d_nodo_label",
                )

            self.assertIsNone(slide.annotation_vote_paths)
            self.assertTrue(slide.annotation_background_only)

    def _build_processor(self, job_dir: str, wsi: _DummyWSI) -> Processor:
        processor = Processor.__new__(Processor)
        processor.job_dir = job_dir
        processor.skip_errors = False
        processor.wsis = [wsi]
        return processor

    def test_run_segmentation_manual_mask_empty_path_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = _DummyWSI(name="slide", ext=".svs", manual_tissue_mask_path="")
            processor = self._build_processor(tmpdir, wsi)

            with self.assertRaises(ValueError):
                processor.run_segmentation_job(segmentation_source="manual_mask")

    def test_run_segmentation_manual_mask_nonexistent_path_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = _DummyWSI(
                name="slide",
                ext=".svs",
                manual_tissue_mask_path=os.path.join(tmpdir, "does_not_exist.zarr"),
            )
            processor = self._build_processor(tmpdir, wsi)

            with self.assertRaises(FileNotFoundError):
                processor.run_segmentation_job(segmentation_source="manual_mask")

    def test_run_segmentation_manual_mask_valid_path_dispatches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "manual_mask.zarr")
            with open(mask_path, "w", encoding="utf-8") as handle:
                handle.write("")

            wsi = _DummyWSI(name="slide", ext=".svs", manual_tissue_mask_path=mask_path)
            processor = self._build_processor(tmpdir, wsi)

            with patch.object(processor_module.gpd, "read_file", return_value=MagicMock(empty=False)):
                out_dir = processor.run_segmentation_job(
                    segmentation_source="manual_mask",
                    manual_mask_source_level=5,
                    manual_mask_target_level=0,
                    manual_mask_tissue_thr=2,
                )

            self.assertEqual(out_dir, os.path.join(tmpdir, "contours"))
            self.assertEqual(len(wsi.manual_calls), 1)
            self.assertEqual(wsi.manual_calls[0]["mask_path"], mask_path)
            self.assertEqual(wsi.manual_calls[0]["source_level"], 5)
            self.assertEqual(wsi.manual_calls[0]["target_level"], 0)
            self.assertEqual(wsi.manual_calls[0]["tissue_thr"], 2)
            self.assertEqual(wsi.release_calls, 1)

    def test_run_segmentation_model_path_still_calls_segment_tissue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wsi = _DummyWSI(name="slide", ext=".svs", manual_tissue_mask_path="")
            processor = self._build_processor(tmpdir, wsi)

            with patch.object(processor_module.gpd, "read_file", return_value=MagicMock(empty=False)):
                out_dir = processor.run_segmentation_job(
                    segmentation_model=object(),
                    seg_mag=10,
                    device="cpu",
                    segmentation_source="model",
                )

            self.assertEqual(out_dir, os.path.join(tmpdir, "contours"))
            self.assertEqual(len(wsi.model_calls), 1)
            self.assertEqual(wsi.model_calls[0]["target_mag"], 10)
            self.assertEqual(wsi.model_calls[0]["device"], "cpu")
            self.assertEqual(len(wsi.manual_calls), 0)


if __name__ == "__main__":
    unittest.main()
