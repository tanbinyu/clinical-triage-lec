from __future__ import annotations

import csv
import io
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from .config import DatasetConfig, PipelineConfig
from .records import (
    MediaLocator, SampleRecord, assign_sample_ids, attach_media_fingerprints, stable_sort,
)


class DatasetError(RuntimeError):
    pass


def _require(path: Path, description: str) -> Path:
    if not path.exists():
        raise DatasetError(f"缺少{description}: {path}")
    return path


def _one(paths: Iterable[Path], description: str) -> Path:
    items = sorted(set(paths))
    if not items:
        raise DatasetError(f"未找到{description}")
    if len(items) > 1:
        raise DatasetError(f"{description}不唯一: " + ", ".join(map(str, items[:8])))
    return items[0]


def _zip_member_map(zf: zipfile.ZipFile) -> dict[str, str]:
    result: dict[str, str] = {}
    duplicates: set[str] = set()
    for member in zf.namelist():
        if member.endswith("/") or member.startswith("__MACOSX/"):
            continue
        key = PurePosixPath(member).name.casefold()
        if key in result:
            duplicates.add(key)
        else:
            result[key] = member
    for key in duplicates:
        result.pop(key, None)
    return result


def _zip_member_for_filename(members: dict[str, str], filename: str) -> str | None:
    requested = PurePosixPath(filename).name.casefold()
    exact = members.get(requested)
    if exact is not None:
        return exact
    candidates = [
        member for key, member in members.items()
        if key == requested or key.rsplit(".", 1)[0] == requested
    ]
    return candidates[0] if len(candidates) == 1 else None


def _read_csv_file(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv_zip(zf: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with zf.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().casefold())


def load_lunghist700(spec: DatasetConfig) -> list[SampleRecord]:
    data_root = spec.root / "LungHist700" / "data"
    if not data_root.exists():
        data_root = spec.root / "data"
    csv_path = _require(data_root / "data.csv", "LungHist700 元数据")
    image_root = _require(data_root / "images", "LungHist700 图像目录")
    label_map = {"aca": 0, "scc": 1, "nor": 2}
    records: list[SampleRecord] = []
    for row in _read_csv_file(csv_path):
        superclass = row["superclass"].strip().casefold()
        subclass = row["subclass"].strip().casefold()
        resolution = row["resolution"].strip().casefold()
        image_id = row["image_id"].strip()
        folder = "nor" if superclass == "nor" else f"{superclass}_{subclass}"
        image = image_root / folder / f"{folder}_{resolution}_{image_id}.jpg"
        _require(image, f"LungHist700 图像 {image_id}")
        records.append(SampleRecord(
            dataset=spec.name,
            patient_id=f"LungHist700_{int(row['patient_id']):03d}",
            target=label_map[superclass],
            label_scope=spec.label_scope,
            locator=MediaLocator("file", image),
            source_id=image.stem,
        ))
    return stable_sort(records)


def load_breakhis(spec: DatasetConfig) -> list[SampleRecord]:
    extracted = list(spec.root.glob("**/histology_slides"))
    records: list[SampleRecord] = []
    if extracted:
        root = extracted[0]
        files = sorted(root.rglob("*.png"))
        for image in files:
            parts = image.parts
            label_name = "malignant" if "malignant" in parts else "benign"
            patient = next((p for p in parts if p.startswith("SOB_")), "")
            if not patient:
                raise DatasetError(f"无法从路径解析 BreakHis patient ID: {image}")
            records.append(SampleRecord(
                spec.name, patient, int(label_name == "malignant"), spec.label_scope,
                MediaLocator("file", image), image.stem,
            ))
        return stable_sort(records)

    archive = _one(spec.root.glob("*.tar.gz"), "BreakHis tar.gz")
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.casefold().endswith(".png"):
                continue
            parts = PurePosixPath(member.name).parts
            label_name = "malignant" if "malignant" in parts else "benign"
            patient = next((p for p in parts if p.startswith("SOB_")), "")
            if not patient:
                raise DatasetError(f"无法从 tar 路径解析 BreakHis patient ID: {member.name}")
            records.append(SampleRecord(
                spec.name, patient, int(label_name == "malignant"), spec.label_scope,
                MediaLocator("tar", archive, member.name), PurePosixPath(member.name).stem,
            ))
    return stable_sort(records)


def load_covid_ct(spec: DatasetConfig) -> list[SampleRecord]:
    pos_meta = _require(spec.root / "COVID-CT-MetaInfo.xlsx", "COVID-CT 阳性元数据")
    neg_meta = _require(spec.root / "NonCOVID-CT-MetaInfo.csv", "COVID-CT 阴性元数据")
    pos_zip = _require(spec.root / "Images-processed" / "CT_COVID.zip", "COVID-CT 阳性图像包")
    neg_zip = _require(spec.root / "Images-processed" / "CT_NonCOVID.zip", "COVID-CT 阴性图像包")
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise DatasetError("读取 COVID-CT xlsx 需要 openpyxl") from exc

    workbook = load_workbook(pos_meta, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    headers = [str(x).strip() for x in next(rows)]
    positive = [dict(zip(headers, row, strict=False)) for row in rows]
    negative = _read_csv_file(neg_meta)
    records: list[SampleRecord] = []
    with zipfile.ZipFile(pos_zip) as zf:
        members = _zip_member_map(zf)
        for row in positive:
            if row.get("File name") in (None, ""):
                continue
            filename = str(row["File name"]).strip()
            member = _zip_member_for_filename(members, filename)
            if member is None:
                raise DatasetError(f"COVID-CT 阳性图像不在压缩包中: {filename}")
            patient = str(row["Patient ID"]).strip()
            records.append(SampleRecord(
                spec.name, f"COVID_{patient.replace(' ', '_')}", 1, spec.label_scope,
                MediaLocator("zip", pos_zip, member), filename,
            ))
    with zipfile.ZipFile(neg_zip) as zf:
        members = _zip_member_map(zf)
        for row in negative:
            filename = row["image name"].strip()
            member = _zip_member_for_filename(members, filename)
            if member is None:
                raise DatasetError(f"COVID-CT 阴性图像不在压缩包中: {filename}")
            patient = row["patient id"].strip()
            records.append(SampleRecord(
                spec.name, f"NonCOVID_{patient}", 0, spec.label_scope,
                MediaLocator("zip", neg_zip, member), filename,
            ))
    return stable_sort(records)


def load_ldfc(spec: DatasetConfig) -> list[SampleRecord]:
    base = spec.root / "BMP_classification"
    _require(base, "LDFC BMP_classification")
    records: list[SampleRecord] = []
    for image in sorted(base.rglob("*.bmp")):
        try:
            original_class = int(image.parent.name)
        except ValueError as exc:
            raise DatasetError(f"LDFC 类别目录必须是 0/1/2: {image.parent}") from exc
        if original_class not in (0, 1, 2):
            raise DatasetError(f"LDFC 未知原始类别 {original_class}: {image}")
        patient_raw = image.stem.split("_", 1)[0]
        records.append(SampleRecord(
            spec.name, f"LDFC_{patient_raw}", int(original_class != 0), spec.label_scope,
            MediaLocator("file", image), image.stem,
        ))
    if not records:
        raise DatasetError(f"LDFC 未找到 BMP: {base}")
    return stable_sort(records)


def _imagechd_labels(xlsx: Path, class_names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise DatasetError("读取 ImageCHD xlsx 需要 openpyxl") from exc
    aliases = {_norm(name): index for index, name in enumerate(class_names)}
    result: dict[str, tuple[int, ...]] = {}
    workbook = load_workbook(xlsx, read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        values = sheet.iter_rows(values_only=True)
        try:
            header_values = next(values)
        except StopIteration:
            continue
        headers = [str(v or "").strip() for v in header_values]
        normalized = [_norm(h) for h in headers]
        id_index = next((i for i, h in enumerate(normalized) if h in {
            "id", "patientid", "caseid", "imageid", "filename", "file", "ctid", "case"
        }), None)
        if id_index is None:
            continue
        disease_columns = {i: aliases[h] for i, h in enumerate(normalized) if h in aliases}
        diagnosis_index = next((i for i, h in enumerate(normalized) if h in {
            "diagnosis", "disease", "diseases", "chd", "type", "label"
        }), None)
        for row in values:
            if id_index >= len(row) or row[id_index] in (None, ""):
                continue
            match = re.search(r"(\d+)", str(row[id_index]))
            if not match:
                continue
            case_id = match.group(1)
            labels: set[int] = set()
            for column, label_index in disease_columns.items():
                value = row[column] if column < len(row) else None
                if _norm(value) not in {"", "0", "false", "no", "none", "normal", "nan"}:
                    labels.add(label_index)
            if diagnosis_index is not None and diagnosis_index < len(row):
                diagnosis = str(row[diagnosis_index] or "")
                for token in re.split(r"[,;/+|\s]+", diagnosis):
                    key = _norm(token)
                    if key in aliases:
                        labels.add(aliases[key])
            result[case_id] = tuple(sorted(labels))
    if not result:
        raise DatasetError(f"无法从 ImageCHD 诊断表解析 case/标签: {xlsx}")
    return result


def load_imagechd(spec: DatasetConfig) -> list[SampleRecord]:
    volumes = [p for p in spec.root.rglob("*.nii.gz") if "_image" in p.name.casefold()]
    if not volumes:
        volumes = [p for p in spec.root.rglob("*.nii") if "_image" in p.name.casefold()]
    if not volumes:
        raise DatasetError(
            "ImageCHD 尚未解开内层 13 个分卷；请先运行 prepare_datasets.py imagechd。"
        )
    spreadsheets = [p for p in spec.root.rglob("*.xlsx") if not p.name.startswith("~$")]
    if not spreadsheets:
        raise DatasetError(f"ImageCHD 缺少诊断 xlsx: {spec.root}")
    parsed: list[tuple[int, Path, dict[str, tuple[int, ...]]]] = []
    for candidate in spreadsheets:
        try:
            candidate_labels = _imagechd_labels(candidate, spec.class_names)
        except DatasetError:
            continue
        parsed.append((len(candidate_labels), candidate, candidate_labels))
    if not parsed:
        raise DatasetError(
            "ImageCHD 的 xlsx 中没有可解析的 case ID + 16 类诊断；"
            f"已检查: {', '.join(map(str, spreadsheets))}"
        )
    _, label_file, labels = max(parsed, key=lambda item: item[0])
    records: list[SampleRecord] = []
    for volume in sorted(volumes):
        match = re.search(r"ct_(\d+)_image", volume.name, flags=re.IGNORECASE)
        if not match:
            continue
        case_id = match.group(1)
        if case_id not in labels:
            raise DatasetError(f"ImageCHD 诊断表缺少 ct_{case_id}: {label_file}")
        records.append(SampleRecord(
            spec.name, f"ImageCHD_{case_id}", labels[case_id], spec.label_scope,
            MediaLocator("nifti", volume), f"ct_{case_id}",
        ))
    if not records:
        raise DatasetError("ImageCHD 未找到命名为 ct_<id>_image.nii[.gz] 的体数据")
    return stable_sort(records)


def load_chestxray14(spec: DatasetConfig) -> list[SampleRecord]:
    archive = _one(spec.root.glob("*.zip"), "ChestX-ray14 zip")
    records: list[SampleRecord] = []
    label_index = {name: i for i, name in enumerate(spec.class_names)}
    with zipfile.ZipFile(archive) as zf:
        csv_members = [n for n in zf.namelist() if PurePosixPath(n).name == "Data_Entry_2017.csv"]
        if len(csv_members) != 1:
            raise DatasetError(f"{archive} 中未唯一找到 Data_Entry_2017.csv")
        rows = _read_csv_zip(zf, csv_members[0])
        members = _zip_member_map(zf)
        for row in rows:
            filename = row["Image Index"].strip()
            member = members.get(filename.casefold())
            if member is None:
                raise DatasetError(f"ChestX-ray14 压缩包缺图像: {filename}")
            findings = [x.strip() for x in row["Finding Labels"].split("|")]
            labels = tuple(sorted(label_index[x] for x in findings if x != "No Finding"))
            unknown = [x for x in findings if x != "No Finding" and x not in label_index]
            if unknown:
                raise DatasetError(f"ChestX-ray14 未知标签 {unknown}: {filename}")
            patient = f"ChestXray14_{int(row['Patient ID']):05d}"
            records.append(SampleRecord(
                spec.name, patient, labels, spec.label_scope,
                MediaLocator("zip", archive, member), filename,
            ))
    return stable_sort(records)


def _cbis_csvs(root: Path) -> list[Path]:
    expected = {
        "mass_case_description_train_set.csv",
        "mass_case_description_test_set.csv",
        "calc_case_description_train_set.csv",
        "calc_case_description_test_set.csv",
    }
    found = {p.name.casefold(): p for p in root.rglob("*.csv")}
    missing = sorted(expected - found.keys())
    if missing:
        raise DatasetError(
            "CBIS-DDSM 缺少官方病理真值 CSV: " + ", ".join(missing) +
            "。仅有 DICOM 不能可靠恢复良恶性标签。"
        )
    return [found[name] for name in sorted(expected)]


def load_cbis_ddsm(spec: DatasetConfig) -> list[SampleRecord]:
    csv_paths = _cbis_csvs(spec.root)
    dicoms = list(spec.root.rglob("*.dcm"))
    if not dicoms:
        raise DatasetError(f"CBIS-DDSM 未找到 DICOM: {spec.root}")
    series_to_dicoms: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in dicoms:
        rel = path.relative_to(spec.root)
        for index, part in enumerate(rel.parts[:-1]):
            if re.match(r"^(Mass|Calc)-(Training|Test)_P_", part, flags=re.IGNORECASE):
                series = rel.parts[index + 1] if index + 1 < len(rel.parts) - 1 else ""
                series_to_dicoms[(_norm(part), _norm(series))].append(path)
                break

    records: list[SampleRecord] = []
    for csv_path in csv_paths:
        for row_index, row in enumerate(_read_csv_file(csv_path), start=2):
            normalized = {_norm(k): v for k, v in row.items()}
            patient = str(normalized.get("patientid", "")).strip()
            pathology = _norm(normalized.get("pathology", ""))
            # The case-description files contain full-mammogram, cropped-lesion
            # and ROI-mask paths.  Classification is lesion-level, so only the
            # cropped lesion image is the unambiguous input.
            image_path = str(normalized.get("croppedimagefilepath", "")).strip().replace("\\", "/")
            if not patient or pathology not in {"benign", "benignwithoutcallback", "malignant"}:
                raise DatasetError(f"CBIS-DDSM 元数据异常: {csv_path}:{row_index}")
            path_parts = PurePosixPath(image_path).parts if image_path else ()
            case_index = next((
                index for index, part in enumerate(path_parts)
                if re.match(r"^(Mass|Calc)-(Training|Test)_P_", part, flags=re.IGNORECASE)
            ), None)
            if case_index is None or case_index + 1 >= len(path_parts):
                raise DatasetError(
                    f"CBIS-DDSM cropped image file path 格式异常: "
                    f"{csv_path}:{row_index} {image_path}"
                )
            key = (_norm(path_parts[case_index]), _norm(path_parts[case_index + 1]))
            matches = series_to_dicoms.get(key, [])
            if len(matches) != 1:
                raise DatasetError(
                    f"CBIS-DDSM cropped lesion series 匹配到 {len(matches)} 个 DICOM: "
                    f"{csv_path}:{row_index} {image_path}"
                )
            dicom = matches[0]
            source_id = f"{csv_path.stem}_{row_index:05d}"
            records.append(SampleRecord(
                spec.name, f"CBIS_{patient}", int(pathology == "malignant"), spec.label_scope,
                MediaLocator("dicom", dicom), source_id,
            ))
    return stable_sort(records)


def load_bus_bra(spec: DatasetConfig) -> list[SampleRecord]:
    archive = _one(spec.root.glob("*.zip"), "BUS-BRA zip")
    records: list[SampleRecord] = []
    with zipfile.ZipFile(archive) as zf:
        metadata = [n for n in zf.namelist() if n.casefold().endswith("/bus_data.csv")]
        if len(metadata) != 1:
            raise DatasetError(f"{archive} 中未唯一找到 bus_data.csv")
        rows = _read_csv_zip(zf, metadata[0])
        members = _zip_member_map(zf)
        for row in rows:
            image_id = row["ID"].strip()
            member = members.get(f"{image_id}.png".casefold())
            if member is None or "/masks/" in member.casefold():
                image_members = [n for n in zf.namelist() if n.casefold().endswith(f"/images/{image_id}.png".casefold())]
                if len(image_members) != 1:
                    raise DatasetError(f"BUS-BRA 图像不唯一或缺失: {image_id}")
                member = image_members[0]
            pathology = row["Pathology"].strip().casefold()
            if pathology not in {"benign", "malignant"}:
                raise DatasetError(f"BUS-BRA 未知 Pathology: {pathology}")
            records.append(SampleRecord(
                spec.name, f"BUSBRA_{int(row['Case']):04d}", int(pathology == "malignant"),
                spec.label_scope, MediaLocator("zip", archive, member), image_id,
            ))
    return stable_sort(records)


def load_pu2756(spec: DatasetConfig) -> list[SampleRecord]:
    image_zip = _require(spec.root / "images.zip", "PU2756 images.zip")
    split_xlsx = _require(spec.root / "five_fold_split.xlsx", "PU2756 标签表")
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise DatasetError("读取 PU2756 xlsx 需要 openpyxl") from exc
    patient_labels: dict[str, int] = {}
    workbook = load_workbook(split_xlsx, read_only=True, data_only=True)
    for sheet in workbook.worksheets:
        rows = sheet.iter_rows(values_only=True)
        try:
            headers = [str(x or "").strip() for x in next(rows)]
        except StopIteration:
            continue
        normalized = [_norm(x) for x in headers]
        try:
            id_index = normalized.index("examinationserialnumber")
            pathology_index = normalized.index("pathologicalresult")
        except ValueError as exc:
            raise DatasetError(f"PU2756 sheet {sheet.title} 缺少患者/病理列") from exc
        for row in rows:
            patient = str(row[id_index] or "").strip()
            pathology = _norm(row[pathology_index])
            if not patient:
                continue
            if pathology not in {"benign", "malignant"}:
                raise DatasetError(f"PU2756 未知病理标签 {row[pathology_index]!r}: {patient}")
            value = int(pathology == "malignant")
            previous = patient_labels.setdefault(patient, value)
            if previous != value:
                raise DatasetError(f"PU2756 同一患者标签冲突: {patient}")

    records: list[SampleRecord] = []
    with zipfile.ZipFile(image_zip) as zf:
        for member in sorted(zf.namelist()):
            path = PurePosixPath(member)
            if member.endswith("/") or path.suffix.casefold() not in {".png", ".jpg", ".jpeg"}:
                continue
            if len(path.parts) < 3:
                continue
            patient = path.parts[-2]
            if patient not in patient_labels:
                raise DatasetError(f"PU2756 标签表缺少患者: {patient}")
            records.append(SampleRecord(
                spec.name, patient, patient_labels[patient], spec.label_scope,
                MediaLocator("zip", image_zip, member), f"{patient}_{path.stem}",
            ))
    return stable_sort(records)


def load_acdc(spec: DatasetConfig) -> list[SampleRecord]:
    group_map = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4, "ARV": 4}
    records: list[SampleRecord] = []
    for folder in sorted(spec.root.glob("**/patient[0-9][0-9][0-9]")):
        info = folder / "Info.cfg"
        if not info.exists():
            continue
        values: dict[str, str] = {}
        for line in info.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
        group = values.get("Group", "").upper()
        if group not in group_map:
            raise DatasetError(f"ACDC 未知 Group {group!r}: {info}")
        try:
            ed, es = int(values["ED"]), int(values["ES"])
        except (KeyError, ValueError) as exc:
            raise DatasetError(f"ACDC Info.cfg 缺少 ED/ES: {info}") from exc
        patient = folder.name
        # CineMA expects its own official resampling/cropping, not raw ACDC grids.
        ed_path = _require(
            spec.prepared_root / patient / f"{patient}_sax_ed.nii.gz",
            "CineMA 官方预处理后的 ACDC ED volume（先运行 prepare）",
        )
        es_path = _require(
            spec.prepared_root / patient / f"{patient}_sax_es.nii.gz",
            "CineMA 官方预处理后的 ACDC ES volume（先运行 prepare）",
        )
        records.append(SampleRecord(
            spec.name, patient, group_map[group], spec.label_scope,
            MediaLocator("acdc_cine", ed_path, extra={"es_path": str(es_path)}), patient,
        ))
    if not records:
        raise DatasetError(f"ACDC 未找到 patientXXX/Info.cfg: {spec.root}")
    return stable_sort(records)


LOADERS: dict[str, Callable[[DatasetConfig], list[SampleRecord]]] = {
    "LungHist700": load_lunghist700,
    "BreakHis": load_breakhis,
    "COVID_CT": load_covid_ct,
    "LDFC": load_ldfc,
    "ImageCHD": load_imagechd,
    "ChestX_ray14": load_chestxray14,
    "CBIS-DDSM": load_cbis_ddsm,
    "BUS-BRA": load_bus_bra,
    "PU2756": load_pu2756,
    "ACDC": load_acdc,
}


def validate_records(spec: DatasetConfig, records: list[SampleRecord]) -> None:
    if not records:
        raise DatasetError(f"{spec.name}: 无可用样本")
    seen: set[str] = set()
    source_ids: set[str] = set()
    sample_ids: set[str] = set()
    patient_targets: dict[str, set[object]] = defaultdict(set)
    for record in records:
        key = record.locator.stable_key
        if key in seen:
            raise DatasetError(f"{spec.name}: 重复媒体定位 {key}")
        seen.add(key)
        if record.source_id in source_ids:
            raise DatasetError(f"{spec.name}: 重复 source_id {record.source_id}")
        source_ids.add(record.source_id)
        if record.sample_id is None or record.sample_id in sample_ids:
            raise DatasetError(f"{spec.name}: sample_id 缺失或重复 {record.sample_id!r}")
        sample_ids.add(record.sample_id)
        if not record.media_fingerprint:
            raise DatasetError(f"{spec.name}/{record.source_id}: 缺少媒体指纹")
        labels = record.target if isinstance(record.target, tuple) else (record.target,)
        if any(label < 0 or label >= spec.num_classes for label in labels):
            raise DatasetError(f"{spec.name}/{record.source_id}: 标签越界 {record.target}")
        patient_targets[record.patient_id].add(record.target)
    if spec.label_scope == "patient":
        conflicts = [p for p, labels in patient_targets.items() if len(labels) > 1]
        if conflicts:
            raise DatasetError(
                f"{spec.name}: patient-level 标签冲突（前 10 个）: {conflicts[:10]}"
            )


def validate_expected_counts(spec: DatasetConfig, records: list[SampleRecord]) -> None:
    patients = len({record.patient_id for record in records})
    problems: list[str] = []
    if spec.expected_samples is not None and len(records) != spec.expected_samples:
        problems.append(f"样本 {len(records)} != 期望 {spec.expected_samples}")
    if spec.expected_patients is not None and patients != spec.expected_patients:
        problems.append(f"患者 {patients} != 期望 {spec.expected_patients}")
    if problems:
        raise DatasetError(f"{spec.name}: 数据完整性检查失败：" + "；".join(problems))


def build_manifest(config: PipelineConfig, names: Iterable[str] | None = None) -> dict[str, list[SampleRecord]]:
    selected = list(names) if names is not None else list(config.datasets)
    manifests: dict[str, list[SampleRecord]] = {}
    for name in selected:
        if name not in config.datasets:
            raise DatasetError(f"未知数据集: {name}")
        records = assign_sample_ids(attach_media_fingerprints(LOADERS[name](config.datasets[name])))
        validate_records(config.datasets[name], records)
        if config.strict_data_validation:
            validate_expected_counts(config.datasets[name], records)
        manifests[name] = records
    return manifests


def summarize_manifest(manifests: dict[str, list[SampleRecord]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, records in manifests.items():
        label_counts: Counter[str] = Counter()
        for record in records:
            if isinstance(record.target, tuple):
                if not record.target:
                    label_counts["normal/none"] += 1
                for label in record.target:
                    label_counts[str(label)] += 1
            else:
                label_counts[str(record.target)] += 1
        rows.append({
            "dataset": name,
            "samples": len(records),
            "patients": len({r.patient_id for r in records}),
            "labels": dict(sorted(label_counts.items())),
        })
    return rows

