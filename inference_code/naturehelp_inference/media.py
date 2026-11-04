from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from .records import MediaLocator, SampleRecord


class MediaError(RuntimeError):
    pass


class MediaReader:
    """Owns archive handles so large ZIP/TAR files are not reopened for every image."""

    def __init__(self, volume_slices: int = 8) -> None:
        self.volume_slices = volume_slices
        self._zips: dict[Path, zipfile.ZipFile] = {}
        self._tars: dict[Path, tarfile.TarFile] = {}

    def close(self) -> None:
        for handle in self._zips.values():
            handle.close()
        for handle in self._tars.values():
            handle.close()
        self._zips.clear()
        self._tars.clear()

    def __enter__(self) -> "MediaReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _pil_open(raw: Any):
        from PIL import Image

        image = Image.open(raw)
        image.load()
        return image.convert("RGB")

    def _read_regular_image(self, locator: MediaLocator):
        if locator.kind == "file":
            return self._pil_open(locator.path)
        if locator.kind == "zip":
            if locator.member is None:
                raise MediaError(f"ZIP locator 缺少 member: {locator.path}")
            archive = self._zips.setdefault(locator.path, zipfile.ZipFile(locator.path))
            return self._pil_open(io.BytesIO(archive.read(locator.member)))
        if locator.kind == "tar":
            if locator.member is None:
                raise MediaError(f"TAR locator 缺少 member: {locator.path}")
            archive = self._tars.setdefault(locator.path, tarfile.open(locator.path, "r:gz"))
            stream = archive.extractfile(locator.member)
            if stream is None:
                raise MediaError(f"TAR member 无法读取: {locator.member}")
            return self._pil_open(io.BytesIO(stream.read()))
        raise MediaError(f"不是普通图像 locator: {locator.kind}")

    @staticmethod
    def _array_to_pil(array: Any):
        import numpy as np
        from PIL import Image

        data = np.asarray(array, dtype=np.float32)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise MediaError("图像不包含有限像素")
        low, high = np.percentile(finite, [1.0, 99.0])
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            scaled = np.zeros(data.shape, dtype=np.uint8)
        else:
            scaled = np.clip((data - low) / (high - low), 0.0, 1.0)
            scaled = (scaled * 255.0).round().astype(np.uint8)
        return Image.fromarray(scaled, mode="L").convert("RGB")

    def _read_dicom(self, path: Path):
        import numpy as np
        try:
            import pydicom
        except ImportError as exc:
            raise MediaError("读取 DICOM 需要 pydicom") from exc
        dataset = pydicom.dcmread(path)
        array = dataset.pixel_array
        try:
            from pydicom.pixels import apply_modality_lut, apply_voi_lut
            array = apply_modality_lut(array, dataset)
            array = apply_voi_lut(array, dataset)
        except (ImportError, ValueError, TypeError):
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            array = array.astype(np.float32) * slope + intercept
        array = np.asarray(array, dtype=np.float32)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            array = array.max() - array
        return self._array_to_pil(array)

    @staticmethod
    def _load_nifti(path: Path):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise MediaError("读取 NIfTI 需要 nibabel") from exc
        data = nib.load(str(path)).get_fdata(dtype="float32")
        while data.ndim > 3:
            data = data[..., 0]
        if data.ndim != 3:
            raise MediaError(f"期望 3D NIfTI，实际 shape={data.shape}: {path}")
        return data

    @staticmethod
    def _slice_indices(depth: int, count: int) -> list[int]:
        import numpy as np

        if depth <= 0:
            raise MediaError("volume depth 必须大于 0")
        n = min(depth, max(1, count))
        if n == 1:
            return [depth // 2]
        edge = (depth - 1) * 0.1
        return sorted(set(int(round(v)) for v in np.linspace(edge, depth - 1 - edge, n)))

    def images_for_record(self, record: SampleRecord, phase2: bool = False) -> list[Any]:
        locator = record.locator
        if locator.kind in {"file", "zip", "tar"}:
            return [self._read_regular_image(locator)]
        if locator.kind == "dicom":
            return [self._read_dicom(locator.path)]
        if locator.kind == "nifti":
            volume = self._load_nifti(locator.path)
            count = self.volume_slices if phase2 else 1
            return [self._array_to_pil(volume[:, :, i]) for i in self._slice_indices(volume.shape[2], count)]
        if locator.kind == "acdc_cine":
            paths = [locator.path]
            es_path = locator.extra.get("es_path")
            if es_path:
                paths.append(Path(es_path))
            images: list[Any] = []
            for path in paths:
                volume = self._load_nifti(path)
                images.append(self._array_to_pil(volume[:, :, volume.shape[2] // 2]))
            return images
        raise MediaError(f"尚未支持 locator kind: {locator.kind}")

    @staticmethod
    def _cinema_pad(volume: Any, shape: tuple[int, int, int]):
        """Match CineMA example preprocessing: ScaleIntensity + end padding."""
        import torch
        import torch.nn.functional as functional

        tensor = torch.as_tensor(volume, dtype=torch.float32)
        if tensor.ndim != 3:
            raise MediaError(f"体数据必须为 3D，实际 {tuple(tensor.shape)}")
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() == 0:
            raise MediaError("CineMA 输入不包含有限像素")
        low, high = finite.min(), finite.max()
        tensor = torch.nan_to_num(tensor, nan=float(low))
        tensor = (tensor - low) / (high - low).clamp_min(1e-6)
        if any(current > target for current, target in zip(tensor.shape, shape, strict=True)):
            raise MediaError(
                f"CineMA 官方预处理体数据大于模型输入 {tuple(tensor.shape)} > {shape}"
            )
        # torch pad order is reversed: z, y, x.  SpatialPadd(method='end')
        # appends zeros and leaves the official LV-centred crop unchanged.
        pads = (
            0, shape[2] - tensor.shape[2],
            0, shape[1] - tensor.shape[1],
            0, shape[0] - tensor.shape[0],
        )
        return functional.pad(tensor, pads).unsqueeze(0)

    def acdc_tensor(self, record: SampleRecord):
        """Return CineMA SAX input [2, 1, 192, 192, 16] for ED and ES."""
        if record.locator.kind != "acdc_cine":
            raise MediaError(f"CineMA 仅支持 acdc_cine，实际 {record.locator.kind}")
        paths = [record.locator.path, Path(record.locator.extra["es_path"])]
        tensors = []
        for path in paths:
            volume = self._load_nifti(path)
            tensors.append(self._cinema_pad(volume, (192, 192, 16)))
        import torch
        return torch.stack(tensors, dim=0)

