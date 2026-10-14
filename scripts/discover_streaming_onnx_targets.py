#!/usr/bin/env python3
"""Discover the ONNX models required by the FunASR streaming server.

Reads runtime defaults and export examples. Does not download weights.
Regenerate docs/agent/streaming_onnx_targets.json from this script; do not
edit that file by hand.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCRIPT = Path("runtime/run_server_2pass.sh")
NAME_MAPS = Path("funasr/download/name_maps_from_hub.py")
EXPORT_ROOT = Path("examples/industrial_data_pretraining")
TARGETS_JSON = Path("docs/agent/streaming_onnx_targets.json")
PACKAGING_DOC = Path("docs/agent/streaming_onnx_packaging.md")

SLOT_ORDER = ("model_dir", "online_model_dir", "vad_dir", "punc_dir")
SLOT_ROLE = {
    "model_dir": "offline_asr",
    "online_model_dir": "online_asr",
    "vad_dir": "vad",
    "punc_dir": "punc",
}
SLOT_ROLE_ZH = {
    "model_dir": "2pass 离线 ASR",
    "online_model_dir": "在线流式 ASR",
    "vad_dir": "VAD",
    "punc_dir": "实时标点",
}
SLOT_EXPORT_META = {
    "model_dir": Path("funasr/models/bicif_paraformer/export_meta.py"),
    "online_model_dir": Path("funasr/models/paraformer_streaming/export_meta.py"),
    "vad_dir": Path("funasr/models/fsmn_vad_streaming/export_meta.py"),
    "punc_dir": Path("funasr/models/ct_transformer_streaming/export_meta.py"),
}
SIDECARS = [
    "config.yaml",
    "am.mvn",
    "tokens.json",
    "vad.yaml",
    "vad.mvn",
    "punc.yaml",
    "seg_dict",
]
SECTION_START = "<!-- streaming-onnx-targets:start -->"
SECTION_END = "<!-- streaming-onnx-targets:end -->"

SLOT_RE = re.compile(
    r"^(?P<slot>model_dir|online_model_dir|vad_dir|punc_dir)"
    r"=(?P<q>['\"]?)(?P<model_id>[^'\"#\s]+)(?P=q)\s*(?:#.*)?$"
)
MODEL_RE = re.compile(r"""model\s*=\s*(?P<q>['"])(?P<model_id>.+?)(?P=q)""")
EXPORT_NAME_RE = re.compile(r"""\.export_name\s*=\s*['"]([A-Za-z0-9_.]+)['"]""")
RETURN_ONNX_RE = re.compile(r"""return\s+['"]([A-Za-z0-9_.]+\.onnx)['"]""")


def stem(model_id: str) -> str:
    """Normalize a hub id so -onnx and -pytorch refer to the same checkpoint."""
    name = model_id.strip().split("/", 1)[-1]
    for suffix in ("-onnx", "-pytorch"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_hub_id(model_id: str) -> bool:
    if "\\" in model_id or model_id.startswith(("/", ".")):
        return False
    org, sep, name = model_id.partition("/")
    return bool(sep and org and name and ":" not in org and "/" not in name)


def read_text(path: Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def runtime_slots() -> dict[str, str]:
    found: dict[str, str] = {}
    for line in read_text(RUNTIME_SCRIPT).splitlines():
        match = SLOT_RE.match(line.strip())
        if match:
            found[match.group("slot")] = match.group("model_id")
    missing = [slot for slot in SLOT_ORDER if slot not in found]
    if missing:
        raise SystemExit(
            f"{RUNTIME_SCRIPT.as_posix()} 缺少槽位: {', '.join(missing)}"
        )
    return found


def export_examples() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for path in sorted((ROOT / EXPORT_ROOT).glob("*/export.py")):
        chosen = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                continue
            match = MODEL_RE.search(line)
            if match and is_hub_id(match.group("model_id")):
                chosen = match.group("model_id")
                break
        if chosen is None:
            continue
        rel = path.relative_to(ROOT).as_posix()
        grouped.setdefault(stem(chosen), []).append(
            {"model_id": chosen, "path": rel}
        )
    return grouped


def load_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    tree = ast.parse(read_text(NAME_MAPS))
    found: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {
                "name_maps_ms",
                "name_maps_hf",
            }:
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise SystemExit(f"{NAME_MAPS.as_posix()} 里的 {target.id} 不是字典")
                found[target.id] = {str(k): str(v) for k, v in value.items()}
    missing = {"name_maps_ms", "name_maps_hf"} - found.keys()
    if missing:
        raise SystemExit(f"{NAME_MAPS.as_posix()} 缺少 {', '.join(sorted(missing))}")
    return found["name_maps_ms"], found["name_maps_hf"]


def fallback_export_id(runtime_id: str) -> str:
    name = runtime_id.split("/", 1)[-1]
    if name.endswith("-onnx"):
        name = name[: -len("-onnx")] + "-pytorch"
    return "iic/" + name


def choose_export_example(candidates: list[dict[str, str]]) -> dict[str, str]:
    def rank(item: dict[str, str]) -> tuple[int, str]:
        org = item["model_id"].split("/", 1)[0]
        return (0 if org == "iic" else 1, item["path"])

    return sorted(candidates, key=rank)[0]


def hf_for_export_id(
    export_id: str, name_maps_ms: dict[str, str], name_maps_hf: dict[str, str]
) -> tuple[str | None, str | None]:
    exact = [
        alias
        for alias, model_id in name_maps_ms.items()
        if model_id == export_id and alias in name_maps_hf
    ]
    stemmed = [
        alias
        for alias, model_id in name_maps_ms.items()
        if stem(model_id) == stem(export_id) and alias in name_maps_hf
    ]
    chosen = sorted(exact or stemmed)
    if not chosen:
        return None, None
    alias = chosen[0]
    return alias, name_maps_hf[alias]


def onnx_filenames(export_meta: Path) -> list[str]:
    text = read_text(export_meta)
    names: list[str] = []
    for match in EXPORT_NAME_RE.finditer(text):
        value = match.group(1)
        filename = value if value.endswith(".onnx") else value + ".onnx"
        if filename not in names:
            names.append(filename)
    if not names:
        for match in RETURN_ONNX_RE.finditer(text):
            filename = match.group(1)
            if filename not in names:
                names.append(filename)
    if not names:
        names = ["model.onnx"]
    names.sort(key=lambda name: (name != "model.onnx", name))
    return names


def export_meta_for(slot: str, export_example: str | None) -> Path:
    if export_example:
        candidate = Path("funasr/models") / Path(export_example).parent.name / "export_meta.py"
        if (ROOT / candidate).is_file():
            return candidate
    return SLOT_EXPORT_META[slot]


def build_manifest() -> dict:
    slots = runtime_slots()
    examples = export_examples()
    name_maps_ms, name_maps_hf = load_name_maps()
    built = []
    for slot in SLOT_ORDER:
        runtime_id = slots[slot]
        runtime_stem = stem(runtime_id)
        export_example = None
        alternatives: list[str] = []
        if runtime_stem in examples:
            chosen = choose_export_example(examples[runtime_stem])
            export_id = chosen["model_id"]
            export_example = chosen["path"]
            resolution = "export_example"
            alternatives = [
                item["path"]
                for item in examples[runtime_stem]
                if item["path"] != export_example
            ]
        else:
            aliases = [
                alias
                for alias, model_id in name_maps_ms.items()
                if stem(model_id) == runtime_stem
            ]
            if aliases:
                alias = sorted(aliases)[0]
                export_id = name_maps_ms[alias]
                resolution = f"name_map:{alias}"
            else:
                export_id = fallback_export_id(runtime_id)
                resolution = "fallback_pytorch_suffix"
        hf_alias, hf_id = hf_for_export_id(export_id, name_maps_ms, name_maps_hf)
        meta = export_meta_for(slot, export_example)
        entry = {
            "slot": slot,
            "role": SLOT_ROLE[slot],
            "role_zh": SLOT_ROLE_ZH[slot],
            "runtime_id": runtime_id,
            "export_id": export_id,
            "resolution": resolution,
            "export_example": export_example,
            "hf_alias": hf_alias,
            "hf_id": hf_id,
            "export_meta": meta.as_posix(),
            "expected_onnx": onnx_filenames(meta),
        }
        if alternatives:
            entry["export_example_alternatives"] = alternatives
        built.append(entry)
    return {
        "version": 1,
        "export_type": "onnx",
        "quantize": False,
        "runtime_script": RUNTIME_SCRIPT.as_posix(),
        "sidecars": list(SIDECARS),
        "slots": built,
    }


def dump_manifest(manifest: dict) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def render_targets_section(manifest: dict) -> str:
    lines = [
        SECTION_START,
        "",
        "### 当前目标",
        "",
        "这一节由 `scripts/discover_streaming_onnx_targets.py` 生成，不要手改。",
        "",
    ]
    for slot in manifest["slots"]:
        lines.append(f"- **{slot['role_zh']}** (`{slot['slot']}` / `{slot['role']}`)")
        lines.append(f"  - 运行时 ID：`{slot['runtime_id']}`")
        lines.append(f"  - 导出 ID：`{slot['export_id']}`")
        lines.append(f"  - 解析：`{slot['resolution']}`")
        if slot.get("export_example"):
            lines.append(f"  - 导出示例：`{slot['export_example']}`")
        if slot.get("hf_id"):
            lines.append(
                f"  - Hugging Face：`{slot['hf_alias']}` → `{slot['hf_id']}`"
            )
        else:
            lines.append("  - Hugging Face：无对应别名，下载失败时不会改走 HF")
        onnx_list = "、".join(f"`{name}`" for name in slot["expected_onnx"])
        lines.append(f"  - ONNX：{onnx_list}")
        lines.append(f"  - 文件名来源：`{slot['export_meta']}`")
        lines.append("")
    lines.append(SECTION_END)
    return "\n".join(lines)


def splice_section(markdown: str, section: str) -> str:
    pattern = re.compile(
        re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END),
        re.DOTALL,
    )
    if not pattern.search(markdown):
        raise SystemExit(
            f"{PACKAGING_DOC.as_posix()} 缺少 {SECTION_START} / {SECTION_END}"
        )
    return pattern.sub(section, markdown)


def github_matrix(manifest: dict) -> dict:
    include = []
    for slot in manifest["slots"]:
        include.append(
            {
                "slot": slot["slot"],
                "role": slot["role"],
                "export_id": slot["export_id"],
                "hf_id": slot["hf_id"] or "",
                "expected_onnx": ",".join(slot["expected_onnx"]),
            }
        )
    return {"include": include}


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def write_outputs(manifest: dict) -> None:
    json_path = ROOT / TARGETS_JSON
    doc_path = ROOT / PACKAGING_DOC
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_bytes(dump_manifest(manifest).encode("utf-8"))
    markdown = normalize_newlines(doc_path.read_text(encoding="utf-8"))
    updated = splice_section(markdown, render_targets_section(manifest))
    if not updated.endswith("\n"):
        updated += "\n"
    doc_path.write_bytes(updated.encode("utf-8"))


def check_outputs(manifest: dict) -> int:
    json_path = ROOT / TARGETS_JSON
    doc_path = ROOT / PACKAGING_DOC
    expected_json = dump_manifest(manifest)
    actual_doc = (
        normalize_newlines(doc_path.read_text(encoding="utf-8"))
        if doc_path.is_file()
        else ""
    )
    if actual_doc and not actual_doc.endswith("\n"):
        actual_doc += "\n"
    expected_doc = splice_section(actual_doc, render_targets_section(manifest))
    if not expected_doc.endswith("\n"):
        expected_doc += "\n"
    problems = []
    actual_json = (
        normalize_newlines(json_path.read_text(encoding="utf-8"))
        if json_path.is_file()
        else ""
    )
    if actual_json != expected_json:
        problems.append(TARGETS_JSON.as_posix())
    if actual_doc != expected_doc:
        problems.append(PACKAGING_DOC.as_posix())
    if problems:
        print(
            "流式 ONNX 清单已过期，先在 dev 上重新生成后再发布:\n"
            "  python scripts/discover_streaming_onnx_targets.py\n"
            "过期文件: " + ", ".join(problems),
            file=sys.stderr,
        )
        return 1
    print("streaming ONNX manifest matches the tree")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 when the committed JSON or the generated doc section is stale",
    )
    parser.add_argument(
        "--github-matrix",
        action="store_true",
        help="print the export job matrix as one JSON object",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    if args.github_matrix:
        print(json.dumps(github_matrix(manifest), ensure_ascii=False))
        return 0
    if args.check:
        return check_outputs(manifest)
    write_outputs(manifest)
    print(f"wrote {TARGETS_JSON.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
