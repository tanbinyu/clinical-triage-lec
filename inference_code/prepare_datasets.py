from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"拒绝解压越界路径: {member.name}")
        handle.extractall(destination, filter="data")


def prepare_breakhis(root: Path) -> None:
    if list(root.glob("**/histology_slides")):
        print("BreakHis 已解压，无需重复处理。")
        return
    archives = list(root.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"期望一个 BreakHis tar.gz，实际 {len(archives)} 个: {root}")
    _safe_extract_tar(archives[0], root)
    print(f"BreakHis 已解压到 {root}")


def _find_7zip() -> Path:
    candidates = [
        shutil.which("7z"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise RuntimeError(
        "ImageCHD 的内层数据是标准分卷 ZIP，需要 7-Zip。"
        "请安装 7-Zip 后重新运行；脚本不会用不可靠的字节拼接代替。"
    )


def prepare_imagechd(root: Path) -> None:
    if list(root.glob("**/ct_*_image.nii.gz")) or list(root.glob("**/ct_*_image.nii")):
        print("ImageCHD 已解压，无需重复处理。")
        return
    outer = root / "ImageCHD.zip"
    if not outer.exists():
        raise RuntimeError(f"缺少 ImageCHD 外层压缩包: {outer}")
    parts_dir = root / "ImageCHD_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer) as archive:
        wanted = [
            name for name in archive.namelist()
            if PureName(name).suffix.casefold() in {f".z{i:02d}" for i in range(1, 100)}
            or name.casefold().endswith(".change2zip")
        ]
        if len(wanted) < 2:
            raise RuntimeError(f"{outer} 中未找到完整的 ImageCHD 分卷")
        for member in wanted:
            destination = parts_dir / Path(member).name
            if not destination.exists() or destination.stat().st_size != archive.getinfo(member).file_size:
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    final_candidates = list(parts_dir.glob("*.change2zip"))
    if len(final_candidates) != 1:
        raise RuntimeError(f"ImageCHD change2zip 分卷不唯一: {final_candidates}")
    final_zip = final_candidates[0].with_suffix(".zip")
    if not final_zip.exists():
        final_candidates[0].replace(final_zip)
    seven_zip = _find_7zip()
    output = root / "ImageCHD_dataset"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(seven_zip), "x", str(final_zip), f"-o{output}", "-y"], check=True)
    if not list(output.glob("**/ct_*_image.nii*")):
        raise RuntimeError(f"7-Zip 返回成功，但未找到 ct_*_image.nii[.gz]: {output}")
    print(f"ImageCHD 已解压到 {output}")


class PureName:
    """Small helper that treats ZIP members as slash-separated names on Windows."""

    def __init__(self, value: str) -> None:
        self.name = value.rsplit("/", 1)[-1]

    @property
    def suffix(self) -> str:
        name = self.name
        return name[name.rfind("."):] if "." in name else ""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Prepare archives that are inefficient/unsupported in-place")
    parser.add_argument("dataset", choices=("breakhis", "imagechd", "all"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Directory containing the source-dataset folders.",
    )
    parser.add_argument(
        "--g-root",
        type=Path,
        dest="data_root",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.dataset in {"breakhis", "all"}:
        prepare_breakhis(args.data_root / "BreakHis")
    if args.dataset in {"imagechd", "all"}:
        prepare_imagechd(args.data_root / "ImageCHD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

