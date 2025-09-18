#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quét cấu trúc dữ liệu VPB:
  batch_[number]/
    audio_chunk/*.wav
    audio_convs/*.mp3
    empty_reason/*.txt
    transcript_edited/*.txt
    transcript_original/*.txt

Tạo NeMo manifest (JSONL) với các field:
  - utt_id: tên file WAV (không kèm đường dẫn)
  - audio_filepath: đường dẫn tuyệt đối tới file WAV
  - text: transcript đã DCD edit (từ transcript_edited)
  - original_text: transcript gốc (model output, từ transcript_original; nếu thiếu => "")
  - duration: độ dài audio (giây, float)

Luật chọn mẫu:
  - Chỉ ghi các audio có transcript_edited tương ứng.
  - Bỏ qua audio chưa gán nhãn (không có transcript_edited) hoặc WAV hỏng (duration<=0).
  - Có thể xuất 1 file gộp hoặc mỗi batch 1 file (tùy tham số).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import contextlib
import wave

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback nếu chưa cài tqdm

# ---------- Helpers ----------

def read_text_file(fp: Path) -> str:
    """
    Đọc file text, chuẩn hóa khoảng trắng 1 dòng.
    """
    for enc in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            txt = fp.read_text(encoding=enc, errors="strict")
            break
        except Exception:
            continue
    else:
        txt = fp.read_text(encoding="utf-8", errors="replace")
    return " ".join(txt.strip().split())


def read_text_if_exists(fp: Optional[Path]) -> str:
    """
    Đọc file nếu tồn tại, ngược lại trả về chuỗi rỗng.
    """
    if not fp or not fp.exists():
        return ""
    try:
        return read_text_file(fp)
    except Exception:
        # Không chặn pipeline chỉ vì lỗi 1 file gốc
        return ""


def wav_duration_seconds(wav_path: Path) -> float:
    """
    Tính duration cho WAV bằng stdlib wave.
    """
    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        frames = wf.getnframes()
        frate = wf.getframerate()
        if frate <= 0:
            return 0.0
        return frames / float(frate)


def pair_paths(batch_dir: Path) -> List[Tuple[Path, Path, Optional[Path]]]:
    """
    Trả về danh sách (audio_wav, transcript_edited_txt, transcript_original_txt|None)
    trong 1 batch_*, chỉ lấy những item có transcript_edited tương ứng.
    """
    audio_dir = batch_dir / "audio_chunk"
    edited_dir = batch_dir / "transcript_edited"
    orig_dir  = batch_dir / "transcript_original"

    if not audio_dir.is_dir():
        return []

    pairs: List[Tuple[Path, Path, Optional[Path]]] = []
    for wav in audio_dir.glob("*.wav"):
        stem = wav.stem  # <audio_name>___<start>___<ch>___<end>
        edited_txt = edited_dir / f"{stem}.txt"
        if edited_txt.exists():
            orig_txt = orig_dir / f"{stem}.txt"
            if not orig_txt.exists():
                orig_txt = None
            pairs.append((wav, edited_txt, orig_txt))
    return pairs


def make_manifest_records(pairs: List[Tuple[Path, Path, Optional[Path]]]) -> List[dict]:
    out = []
    for wav_path, edited_txt_path, orig_txt_path in tqdm(pairs, desc="Building records"):
        try:
            text = read_text_file(edited_txt_path)
            if not text:
                # Bỏ qua mẫu text rỗng (edited)
                continue
            duration = wav_duration_seconds(wav_path)
            if duration <= 0.0:
                # Bỏ qua file hỏng/không hợp lệ
                continue

            original_text = read_text_if_exists(orig_txt_path)

            rec = {
                "utt_id": wav_path.name,                       # dùng tên file wav
                "audio_filepath": str(wav_path.resolve()),     # đường dẫn tuyệt đối
                "text": text,                                  # transcript đã DCD edit
                "original_text": original_text,                # transcript gốc (có thể rỗng)
                "duration": round(float(duration), 4),         # làm tròn 4 chữ số thập phân
            }
            out.append(rec)
        except Exception as e:
            print(f"[WARN] Skip {wav_path.name}: {e}", file=sys.stderr)
    return out


def write_jsonl(items: List[dict], out_fp: Path) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with out_fp.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Tạo NeMo manifest từ folder batch_* VPB.")
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Thư mục gốc chứa các 'batch_*' (vd: /mnt/efs/work/vpb_audio_label)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Đường dẫn file JSONL manifest gộp hoặc folder chứa manifest mỗi batch (tùy cờ --per-batch).",
    )
    ap.add_argument(
        "--per-batch",
        action="store_true",
        help="Nếu set, sẽ xuất mỗi batch một file manifest JSONL trong thư mục --out. "
             "Nếu không set, sẽ gộp tất cả vào 1 file JSONL tại --out.",
    )
    args = ap.parse_args()

    data_root: Path = args.data_root
    if not data_root.is_dir():
        ap.error(f"--data-root không tồn tại hoặc không phải thư mục: {data_root}")

    batch_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("batch_")])
    if not batch_dirs:
        ap.error(f"Không tìm thấy thư mục batch_* trong: {data_root}")

    if args.per_batch:
        out_dir = args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        for bdir in batch_dirs:
            pairs = pair_paths(bdir)
            records = make_manifest_records(pairs)
            out_fp = out_dir / f"{bdir.name}.jsonl"
            write_jsonl(records, out_fp)
            print(f"[OK] {bdir.name}: {len(records)} items -> {out_fp}")
            total += len(records)
        print(f"[DONE] Tổng số mẫu ghi: {total}")
    else:
        all_pairs: List[Tuple[Path, Path, Optional[Path]]] = []
        for bdir in batch_dirs:
            all_pairs.extend(pair_paths(bdir))
        records = make_manifest_records(all_pairs)
        out_fp = args.out
        write_jsonl(records, out_fp)
        print(f"[DONE] Ghi {len(records)} items -> {out_fp}")


if __name__ == "__main__":
    main()
