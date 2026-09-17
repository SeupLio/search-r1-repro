# -*- coding: utf-8 -*-
"""从 ModelScope 下载模型快照到本地目录，供 transformers 直接 from_pretrained 加载。

为什么需要它：原仓库依赖 HuggingFace Hub，但在国内 / 受限网络下
huggingface.co 常常不可达（HTTP 000）。ModelScope 是国内可达的等价源，
且模型文件格式与 HF 完全一致，下载后用 `AutoModelForCausalLM.from_pretrained(<dir>)`
直接可用，无需改任何加载代码。

用法：
    python scripts/download_model_modelscope.py --repo qwen/Qwen2.5-0.5B-Instruct \
        --out models/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={rev}"
RAW = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision={rev}&FilePath={path}"

# 推理/训练必需的最小文件集；其余（README/LICENSE 等）跳过以节省时间
NEED = ("config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json",
        "sentencepiece.bpe.model", "chat_template.jinja")
WEIGHT = ("model.safetensors", "pytorch_model.bin", "mp_model.bin")


def list_files(repo: str, rev: str):
    url = API.format(repo=repo, rev=rev)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    if data.get("Code") != 200:
        raise RuntimeError(f"ModelScope API 返回异常: {data}")
    return data["Data"]["Files"]


def download(repo: str, rev: str, path: str, dst: str):
    url = RAW.format(repo=repo, rev=rev, path=path)
    req = urllib.request.Request(url, headers={"User-Agent": "search-r1-repro"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dst, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"\r    {path}  {total/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="ModelScope 仓库名，如 qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", required=True, help="本地输出目录")
    ap.add_argument("--rev", default="master")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = list_files(args.repo, args.rev)
    picked = [f for f in files
              if f["Name"] in NEED or f["Name"] in WEIGHT
              or f["Name"].endswith(".safetensors")]
    if not picked:
        print("未找到可识别的模型文件，仓库内容：",
              [f["Name"] for f in files][:20], file=sys.stderr)
        return 1

    for f in picked:
        dst = os.path.join(args.out, f["Name"])
        size_mb = f["Size"] / 1e6
        if os.path.exists(dst) and abs(os.path.getsize(dst) - f["Size"]) < 1024:
            print(f"  跳过（已存在）{f['Name']} {size_mb:.1f} MB")
            continue
        print(f"  下载 {f['Name']}  {size_mb:.1f} MB")
        try:
            download(args.repo, args.rev, f["Name"], dst)
        except Exception as e:
            print(f"  !! 失败 {f['Name']}: {e}")
            return 1
    print(f"\n完成 -> {os.path.abspath(args.out)}")
    print("加载方式：")
    print(f"  AutoModelForCausalLM.from_pretrained(r'{os.path.abspath(args.out)}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
