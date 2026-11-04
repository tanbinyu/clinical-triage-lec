from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from naturehelp_inference.config import DATASET_ORDER, DISEASE_TO_DATASETS, load_config
from naturehelp_inference.output import (
    atomic_write_json, publish_staged_outputs, validate_payload, validate_result_set,
)
from naturehelp_inference.manifest import manifest_digest, read_manifest, write_manifest
from naturehelp_inference.records import (
    MediaLocator, SampleRecord, assign_sample_ids,
)
from naturehelp_inference.progress import ProgressStore
from naturehelp_inference.splits import (
    make_downstream_partition, make_patient_stratified_split, records_for_split,
)


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(Path(__file__).parents[1] / "configs" / "default.toml")

    @staticmethod
    def record(patient: str, source: str) -> SampleRecord:
        return SampleRecord(
            dataset="BreakHis",
            patient_id=patient,
            target=1,
            label_scope="patient",
            locator=MediaLocator("file", Path("fixtures") / f"{source}.png"),
            source_id=source,
        )

    def test_disease_mapping_is_exact(self) -> None:
        self.assertEqual(
            tuple(DISEASE_TO_DATASETS[index][0] for index in range(10)),
            DATASET_ORDER,
        )
        self.assertEqual(DATASET_ORDER[3], "LDFC")
        self.assertEqual(self.config.datasets["ImageCHD"].num_classes, 16)

    def test_phase_json_has_exact_fields(self) -> None:
        payload = {
            "sample_ACDC_00000": {
                "uncertainty": 0.25,
                "correctness": 1,
                "predicted": 2,
                "true": 2,
                "patient_id": "patient001",
                "label_scope": "patient",
                "inference_model": "CineMA/ACDC_adapter",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_2_ACDC.json"
            atomic_write_json(path, payload, phase=2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_extra_json_field_is_rejected(self) -> None:
        payload = {
            "sample_ACDC_00000": {
                "uncertainty": 0.25,
                "correctness": 1,
                "predicted": 2,
                "true": 2,
                "patient_id": "patient001",
                "label_scope": "patient",
                "unexpected_extra_field": True,
            }
        }
        with self.assertRaises(ValueError):
            validate_payload(payload, phase=2)

    def test_staged_publish_archives_previous_and_stale_outputs(self) -> None:
        payload = {
            "sample_ACDC_00000": {
                "uncertainty": 0.25,
                "correctness": 1,
                "predicted": 2,
                "true": 2,
                "patient_id": "patient001",
                "label_scope": "patient",
                "inference_model": "CineMA/ACDC_adapter",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging, output = root / "staging", root / "output"
            staging.mkdir()
            output.mkdir()
            name = "phase_2_ACDC.json"
            atomic_write_json(staging / name, payload, phase=2)
            (output / name).write_text("{}", encoding="utf-8")
            (output / "phase_2_LIDC_IDRI.json").write_text("{}", encoding="utf-8")
            archived = publish_staged_outputs(
                staging, output, {name}, archive_unexpected=True
            )
            self.assertEqual(len(archived), 2)
            self.assertEqual(json.loads((output / name).read_text(encoding="utf-8")), payload)
            self.assertFalse((output / "phase_2_LIDC_IDRI.json").exists())

    def test_result_set_cross_checks_patient_and_sample_ids(self) -> None:
        phase1 = {
            "sample_ACDC_00000": {
                "uncertainty": 0.1,
                "correctness": 1,
                "predicted_disease": 9,
                "true_disease": 9,
                "source_dataset": "ACDC",
                "patient_id": "patient001",
            }
        }
        phase2 = {
            "sample_ACDC_00000": {
                "uncertainty": 0.2,
                "correctness": 1,
                "predicted": 0,
                "true": 0,
                "patient_id": "patient001",
                "label_scope": "patient",
                "inference_model": "CineMA/ACDC_adapter",
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_write_json(root / "phase_1.json", phase1, phase=1)
            atomic_write_json(root / "phase_2_ACDC.json", phase2, phase=2)
            validate_result_set(root, {"ACDC": 9})
            phase2["sample_ACDC_00000"]["patient_id"] = "patient999"
            atomic_write_json(root / "phase_2_ACDC.json", phase2, phase=2)
            with self.assertRaises(ValueError):
                validate_result_set(root, {"ACDC": 9})

    def test_progress_store_is_resumable_and_identity_bound(self) -> None:
        item = {
            "uncertainty": 0.2,
            "correctness": 1,
            "predicted": 0,
            "true": 0,
            "patient_id": "patient001",
            "label_scope": "patient",
            "inference_model": "CineMA/ACDC_adapter",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.sqlite"
            with ProgressStore(path, "run-a", phase=2) as progress:
                progress.put_many({"sample_ACDC_00000": item})
            with ProgressStore(path, "run-a", phase=2) as progress:
                self.assertEqual(progress.load()["sample_ACDC_00000"], item)
            with self.assertRaises(RuntimeError):
                ProgressStore(path, "run-b", phase=2)

    def test_sample_ids_are_stable_before_subsetting(self) -> None:
        records = assign_sample_ids([
            self.record("P002", "c"), self.record("P001", "b"), self.record("P001", "a")
        ])
        mapping = {record.source_id: record.sample_id for record in records}
        subset = [record for record in records if record.patient_id == "P001"]
        self.assertEqual([record.sample_id for record in subset], [mapping["a"], mapping["b"]])

    def test_manifest_roundtrip_preserves_digest(self) -> None:
        records = assign_sample_ids([self.record("P001", "a"), self.record("P002", "b")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BreakHis.jsonl"
            write_manifest(path, records)
            loaded = read_manifest(path)
        self.assertEqual(manifest_digest(records), manifest_digest(loaded))

    def test_stratified_patient_split_is_disjoint_and_balanced(self) -> None:
        spec = self.config.datasets["ACDC"]
        records = []
        for index in range(100):
            records.append(SampleRecord(
                dataset="ACDC",
                patient_id=f"patient{index:03d}",
                target=index % 5,
                label_scope="patient",
                locator=MediaLocator("file", Path("fixtures") / f"{index}.png"),
                source_id=str(index),
                sample_id=f"sample_ACDC_{index:05d}",
            ))
        assignments = make_patient_stratified_split(records, spec, 123, (0.35, 0.05, 0.60))
        patient_sets = [
            {record.patient_id for record in records_for_split(records, assignments, split)}
            for split in ("train", "val", "downstream")
        ]
        self.assertEqual([len(value) for value in patient_sets], [35, 5, 60])
        self.assertTrue(patient_sets[0].isdisjoint(patient_sets[1] | patient_sets[2]))
        for split in ("train", "val", "downstream"):
            labels = [record.target for record in records_for_split(records, assignments, split)]
            self.assertEqual(len(set(labels)), 5)

    def test_paper_split_uses_largest_remainder_patient_counts(self) -> None:
        expected = {
            45: (16, 2, 11, 8, 8),
            82: (29, 4, 19, 15, 15),
            271: (95, 13, 65, 49, 49),
            86: (30, 4, 21, 16, 15),
            110: (39, 5, 26, 20, 20),
            30805: (10782, 1540, 7393, 5545, 5545),
            1566: (548, 78, 376, 282, 282),
            1064: (372, 53, 255, 192, 192),
            2756: (965, 138, 661, 496, 496),
            150: (53, 7, 36, 27, 27),
        }
        from naturehelp_inference.splits import _capacities, _largest_remainder
        for patients, counts in expected.items():
            self.assertEqual(
                tuple(_largest_remainder(
                    patients, (0.35, 0.05, 0.24, 0.18, 0.18)
                )),
                counts,
            )
        downstream = [sum(counts[2:]) for counts in expected.values()]
        source_balanced = [
            _capacities(min(2000, count), (0.40, 0.30, 0.30))
            for count in downstream
        ]
        self.assertEqual(sum(min(2000, count) for count in downstream), 5679)
        self.assertEqual(
            [sum(row[index] for row in source_balanced) for index in range(3)],
            [2270, 1705, 1704],
        )


if __name__ == "__main__":
    unittest.main()

