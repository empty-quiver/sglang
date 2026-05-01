#!/usr/bin/env python3
"""Convert AutoRound/GPTQ MoE expert tensors to kt-kernel RAWINT4 layout.

This is intended for heterogeneous KT serving where SGLang uses the original
AutoRound checkpoint for GPU experts, while kt-kernel loads CPU experts from a
RAWINT4-compatible expert-only directory.

Input expert tensors are expected in symmetric GPTQ/AutoRound form:
  *.{gate,up,down}_proj.qweight: int32 [K/8, N]
  *.{gate,up,down}_proj.scales:  fp16/bf16 [K/group_size, N]

Output tensors use kt-kernel RAWINT4 names and layout:
  *.{gate,up,down}_proj.weight_packed: uint8 [N, K/2]
  *.{gate,up,down}_proj.weight_scale:  bf16  [N, K/group_size]

The conversion is a repack/transpose only. It does not re-quantize weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def gptq_qweight_to_rawint4(qweight: torch.Tensor) -> torch.Tensor:
    """Convert GPTQ [K/8, N] int32 packing to RAWINT4 [N, K/2] bytes."""
    if qweight.dtype != torch.int32:
        raise TypeError(f"expected int32 qweight, got {qweight.dtype}")
    if qweight.dim() != 2:
        raise ValueError(f"expected 2D qweight, got shape={tuple(qweight.shape)}")

    k_packed, n = qweight.shape
    qbytes = qweight.contiguous().view(torch.uint8).reshape(k_packed, n, 4)
    return qbytes.permute(1, 0, 2).reshape(n, k_packed * 4).contiguous()


def gptq_scales_to_rawint4(scales: torch.Tensor) -> torch.Tensor:
    """Convert GPTQ [K/group, N] scales to RAWINT4 [N, K/group] BF16."""
    if scales.dim() != 2:
        raise ValueError(f"expected 2D scales, got shape={tuple(scales.shape)}")
    return scales.t().contiguous().to(torch.bfloat16)


def convert_key(key: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
    if ".mlp.experts." not in key:
        return None

    for proj in PROJECTIONS:
        q_suffix = f".{proj}.qweight"
        s_suffix = f".{proj}.scales"
        if key.endswith(q_suffix):
            return key[: -len(q_suffix)] + f".{proj}.weight_packed", gptq_qweight_to_rawint4(tensor)
        if key.endswith(s_suffix):
            return key[: -len(s_suffix)] + f".{proj}.weight_scale", gptq_scales_to_rawint4(tensor)

    return None


def load_index(model_dir: Path) -> Dict[str, str] | None:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    with index_path.open() as f:
        return json.load(f).get("weight_map", {})


def copy_metadata_files(src: Path, dst: Path) -> None:
    for name in ("config.json", "quantization_config.json"):
        src_path = src / name
        if src_path.exists():
            shutil.copy2(src_path, dst / name)


def convert_file(src_file: Path, dst_file: Path, skip_existing: bool) -> tuple[int, int]:
    if skip_existing and dst_file.exists():
        return 0, 1

    converted: Dict[str, torch.Tensor] = {}
    with safe_open(src_file, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if ".mlp.experts." not in key:
                continue
            if not any(
                key.endswith(f".{proj}.qweight") or key.endswith(f".{proj}.scales")
                for proj in PROJECTIONS
            ):
                continue
            out = convert_key(key, reader.get_tensor(key))
            if out is not None:
                out_key, out_tensor = out
                converted[out_key] = out_tensor

    if not converted:
        return 0, 0

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, dst_file)
    return len(converted), 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(model_dir, output_dir)

    index = load_index(model_dir)
    if index is None:
        shard_names = sorted(p.name for p in model_dir.glob("*.safetensors"))
    else:
        shard_names = sorted(set(index.values()))

    total_tensors = 0
    skipped_files = 0
    written_files = 0
    for shard_name in shard_names:
        src_file = model_dir / shard_name
        if not src_file.exists():
            continue
        dst_file = output_dir / shard_name
        count, skipped = convert_file(src_file, dst_file, skip_existing=not args.overwrite)
        skipped_files += skipped
        if count:
            written_files += 1
            total_tensors += count
            print(f"converted {src_file.name}: {count} tensors -> {dst_file}", flush=True)

    manifest = {
        "format": "kt-rawint4-from-autoround-gptq",
        "source": str(model_dir),
        "written_files": written_files,
        "skipped_files": skipped_files,
        "converted_tensors": total_tensors,
    }
    with (output_dir / "rawint4_conversion_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
