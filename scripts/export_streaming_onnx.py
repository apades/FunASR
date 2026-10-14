#!/usr/bin/env python3
"""Export one streaming slot to ONNX and zip it for a GitHub Release.

Uses the same call as the streaming examples:
AutoModel.export(type="onnx", quantize=False). The slot list comes from
docs/agent/streaming_onnx_targets.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("docs/agent/streaming_onnx_targets.json")
WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}


def load_slot(manifest_path: Path, slot_name: str) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for slot in manifest["slots"]:
        if slot["slot"] == slot_name:
            return manifest, slot
    known = ", ".join(slot["slot"] for slot in manifest["slots"])
    raise SystemExit(f"未知槽位 {slot_name}。清单里有: {known}")


def load_automodel(model_id: str, hub: str):
    from funasr import AutoModel

    return AutoModel(
        model=model_id,
        device="cpu",
        hub=hub,
        disable_update=True,
        disable_pbar=True,
    )


def acquire_model(slot: dict):
    export_id = slot["export_id"]
    errors = []
    try:
        return load_automodel(export_id, "ms")
    except Exception as exc:
        errors.append(f"hub=ms model={export_id}: {exc}")
    hf_id = slot.get("hf_id")
    if hf_id:
        try:
            return load_automodel(hf_id, "hf")
        except Exception as exc:
            errors.append(f"hub=hf model={hf_id}: {exc}")
    raise SystemExit("导出源下载失败:\n" + "\n".join(errors))


def place_expected_onnx(slot_dir: Path, expected: list[str]) -> None:
    missing = []
    for name in expected:
        direct = slot_dir / name
        if direct.is_file():
            continue
        matches = [path for path in slot_dir.rglob(name) if path.is_file()]
        if len(matches) != 1:
            missing.append(name)
            continue
        shutil.copy2(matches[0], direct)
    if missing:
        found = sorted(path.name for path in slot_dir.rglob("*.onnx"))
        raise SystemExit(
            "导出结果缺少 " + ", ".join(missing) + "。目录里的 onnx: " + ", ".join(found)
        )


def copy_sidecars(checkpoint: Path, slot_dir: Path, names: list[str]) -> None:
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint 目录不存在: {checkpoint}")
    for name in names:
        source = checkpoint / name
        if not source.is_file():
            print(f"sidecar absent: {name}")
            continue
        if source.suffix.lower() in WEIGHT_SUFFIXES:
            print(f"sidecar skipped: {name}")
            continue
        shutil.copy2(source, slot_dir / name)
        print(f"sidecar copied: {name}")


def drop_weight_files(slot_dir: Path) -> None:
    for path in slot_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
            path.unlink()
            print(f"removed weight file: {path.name}")


def zip_directory(slot_dir: Path, zip_path: Path, role: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(slot_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in WEIGHT_SUFFIXES:
                continue
            arcname = (Path(role) / path.relative_to(slot_dir)).as_posix()
            archive.write(path, arcname)
    print(f"wrote {zip_path}")


def export_slot(slot: dict, manifest: dict, output_dir: Path) -> Path:
    role = slot["role"]
    slot_dir = output_dir / role
    if slot_dir.exists():
        shutil.rmtree(slot_dir)
    slot_dir.mkdir(parents=True)
    model = acquire_model(slot)
    model.export(type="onnx", quantize=False, output_dir=str(slot_dir))
    place_expected_onnx(slot_dir, slot["expected_onnx"])
    checkpoint = getattr(model, "model_path", None)
    if not checkpoint:
        raise SystemExit("AutoModel 没有返回 model_path，无法复制运行时随文件")
    copy_sidecars(Path(checkpoint), slot_dir, manifest["sidecars"])
    drop_weight_files(slot_dir)
    zip_path = output_dir / f"{role}.zip"
    zip_directory(slot_dir, zip_path, role)
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True, help="runtime slot name, such as online_model_dir")
    parser.add_argument("--output-dir", default="export/streaming-onnx")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest, slot = load_slot(manifest_path, args.slot)
    if manifest.get("export_type") != "onnx" or manifest.get("quantize") is not False:
        raise SystemExit("清单里的导出类型必须是 onnx 且 quantize 为 false")
    export_slot(slot, manifest, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
