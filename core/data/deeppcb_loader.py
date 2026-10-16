"""DeepPCB dataset loader, annotation parser, and zero-leakage SHA-256 split verifier.

Dataset structure (`/home/tofui/project/pcb/data/DeepPCB/PCBData`):
- `trainval.txt` (1,000 image/annotation entries)
- `test.txt` (500 image/annotation entries)
- Each manifest line: `<group>/<sub_dir>/<id>.jpg <group>/<sub_dir>_not/<id>.txt`
- Actual image files on disk: `<id>_temp.jpg` (golden template) and `<id>_test.jpg` (tested PCB)
- Each annotation line: `x1 y1 x2 y2 class_id` where `class_id` is in `{1..6}`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from core.agent.schemas import BoundingBoxDetection, DefectClass

DEFAULT_DEEPPCB_ROOT = Path("/home/tofui/project/pcb/data/DeepPCB/PCBData")

DEEPPCB_CLASS_MAP: dict[int, DefectClass] = {
    1: "Open",
    2: "Short",
    3: "Mousebite",
    4: "Spur",
    5: "Spurious Copper",
    6: "Pin-hole",
}


@dataclass(frozen=True, slots=True)
class DeepPCBSample:
    """Metadata and resolved file paths for a single DeepPCB template/test/annotation pair."""

    sample_id: str
    split: Literal["trainval", "test"]
    template_path: Path
    test_path: Path
    annotation_path: Path


def parse_annotation_file(txt_path: str | Path) -> list[BoundingBoxDetection]:
    """Parse a DeepPCB annotation `.txt` file into validated `BoundingBoxDetection` objects.

    Each line in a DeepPCB `.txt` file has 5 space-separated integers:
        `x1 y1 x2 y2 class_id`
    Ground-truth bounding boxes are assigned `confidence=1.0`.
    """
    path = Path(txt_path)
    if not path.is_file():
        raise FileNotFoundError(f"DeepPCB annotation file not found: {path}")

    detections: list[BoundingBoxDetection] = []
    raw_text = path.read_text(encoding="utf-8")

    for line_num, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"Malformed annotation at {path}:{line_num} (expected 5 fields, got {len(parts)}): {raw_line!r}"
            )

        try:
            x1, y1, x2, y2, class_id = (int(tok) for tok in parts)
        except ValueError as exc:
            raise ValueError(
                f"Non-integer token in annotation at {path}:{line_num}: {raw_line!r}"
            ) from exc

        if class_id not in DEEPPCB_CLASS_MAP:
            raise ValueError(
                f"Unknown DeepPCB class_id={class_id} at {path}:{line_num}. "
                f"Valid IDs: {sorted(DEEPPCB_CLASS_MAP.keys())}"
            )

        detections.append(
            BoundingBoxDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                defect_class=DEEPPCB_CLASS_MAP[class_id],
                confidence=1.0,
            )
        )

    return detections


class DeepPCBDataset:
    """Deterministic loader and zero-leakage auditor for the DeepPCB dataset."""

    def __init__(self, data_root: str | Path = DEFAULT_DEEPPCB_ROOT) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"DeepPCB dataset root not found: {self.data_root}")

        self.trainval_manifest = self.data_root / "trainval.txt"
        self.test_manifest = self.data_root / "test.txt"

        self.trainval_samples: list[DeepPCBSample] = self._parse_manifest(
            self.trainval_manifest, split="trainval"
        )
        self.test_samples: list[DeepPCBSample] = self._parse_manifest(
            self.test_manifest, split="test"
        )

    def _parse_manifest(
        self,
        manifest_path: Path,
        split: Literal["trainval", "test"],
    ) -> list[DeepPCBSample]:
        """Parse `trainval.txt` or `test.txt` into resolved `DeepPCBSample` records."""
        if not manifest_path.is_file():
            raise FileNotFoundError(f"DeepPCB split manifest not found: {manifest_path}")

        samples: list[DeepPCBSample] = []
        for line_num, raw_line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed manifest entry at {manifest_path}:{line_num}: {raw_line!r}"
                )

            img_rel, ann_rel = parts
            if not img_rel.endswith(".jpg"):
                raise ValueError(
                    f"Expected '.jpg' base path in {manifest_path}:{line_num}, got {img_rel!r}"
                )

            stem_rel = img_rel[:-4]
            sample_id = Path(stem_rel).name
            template_path = self.data_root / f"{stem_rel}_temp.jpg"
            test_path = self.data_root / f"{stem_rel}_test.jpg"
            annotation_path = self.data_root / ann_rel

            if not template_path.is_file():
                raise FileNotFoundError(f"Missing template image: {template_path}")
            if not test_path.is_file():
                raise FileNotFoundError(f"Missing tested image: {test_path}")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

            samples.append(
                DeepPCBSample(
                    sample_id=sample_id,
                    split=split,
                    template_path=template_path,
                    test_path=test_path,
                    annotation_path=annotation_path,
                )
            )

        return samples

    def get_split(self, split: Literal["trainval", "test"]) -> list[DeepPCBSample]:
        """Return all samples for the requested split (`'trainval'` or `'test'`)."""
        if split == "trainval":
            return self.trainval_samples
        if split == "test":
            return self.test_samples
        raise ValueError(f"Invalid split {split!r}; expected 'trainval' or 'test'.")

    def load_sample(
        self,
        sample: DeepPCBSample | int,
        split: Literal["trainval", "test"] = "test",
    ) -> tuple[np.ndarray, np.ndarray, list[BoundingBoxDetection]]:
        """Load `(template_img, test_img, ground_truth_boxes)` for a sample or index."""
        if isinstance(sample, int):
            target = self.get_split(split)[sample]
        else:
            target = sample

        template_img = cv2.imread(str(target.template_path), cv2.IMREAD_GRAYSCALE)
        test_img = cv2.imread(str(target.test_path), cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            raise ValueError(f"Failed to decode template image: {target.template_path}")
        if test_img is None:
            raise ValueError(f"Failed to decode test image: {target.test_path}")

        boxes = parse_annotation_file(target.annotation_path)
        return template_img, test_img, boxes

    def verify_zero_leakage(self) -> dict[str, object]:
        """Verify disjoint `trainval` and `test` splits and return deterministic SHA-256 hashes."""
        trainval_ids = {s.sample_id for s in self.trainval_samples}
        test_ids = {s.sample_id for s in self.test_samples}

        trainval_paths = {s.template_path.resolve() for s in self.trainval_samples} | {
            s.test_path.resolve() for s in self.trainval_samples
        }
        test_paths = {s.template_path.resolve() for s in self.test_samples} | {
            s.test_path.resolve() for s in self.test_samples
        }

        overlapping_ids = sorted(trainval_ids & test_ids)
        overlapping_paths = sorted(str(p) for p in (trainval_paths & test_paths))
        overlap_count = len(overlapping_ids) + len(overlapping_paths)

        trainval_canonical = "\n".join(sorted(trainval_ids)).encode("utf-8")
        test_canonical = "\n".join(sorted(test_ids)).encode("utf-8")

        trainval_sha256 = hashlib.sha256(trainval_canonical).hexdigest()
        test_sha256 = hashlib.sha256(test_canonical).hexdigest()

        if overlap_count > 0:
            raise ValueError(
                f"Data leakage detected between trainval and test splits! "
                f"Overlapping sample IDs: {overlapping_ids}"
            )

        return {
            "zero_leakage": True,
            "overlap_count": 0,
            "trainval_count": len(self.trainval_samples),
            "test_count": len(self.test_samples),
            "total_count": len(self.trainval_samples) + len(self.test_samples),
            "trainval_sha256": trainval_sha256,
            "test_sha256": test_sha256,
        }
