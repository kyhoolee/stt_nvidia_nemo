# freeze_utils.py
from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Dict, Any, Optional, Union
import torch
import torch.nn as nn

from lightning.pytorch.callbacks import Callback

# ----------------------------------
# Freeze scheduler callback
# ----------------------------------

class UnfreezeAtEpoch(Callback):
    def __init__(self, epoch_to_unfreeze: int):
        super().__init__()
        self.epoch_to_unfreeze = epoch_to_unfreeze
        self._done = False

    def on_train_epoch_start(self, trainer, pl_module):
        if self._done:
            return
        current = trainer.current_epoch or 0
        if self.epoch_to_unfreeze >= 0 and current >= self.epoch_to_unfreeze:
            print(f"🔓 Unfreezing all params at epoch {current} …")
            unfreeze_all(pl_module)
            self._done = True




def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _parse_stage_str(s: str) -> Dict[str, Any]:
    """
    e=0,enc_bottom_k=12,pre=1,subs=1,pos=1,dec_all=1,joint=1,prefix=encoder.pre_encode.out|decoder.prediction.embed
    """
    out: Dict[str, Any] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Malformed stage token: {pair}")
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "e":
            out["e"] = int(v)
        elif k in {"enc_bottom_k", "enc_top_k"}:
            out[k] = int(v)
        elif k == "enc_bottom_ratio":
            out[k] = float(v)
        elif k in {"pre", "subs", "pos", "spec", "dec_all", "dec_embed", "dec_rnn", "joint", "unfreeze_all"}:
            out[k] = _parse_bool(v)
        elif k in {"prefix", "regex", "types"}:
            out[k] = [t for t in v.split("|") if t]
        else:
            raise ValueError(f"Unknown stage key: {k}")
    if "e" not in out:
        raise ValueError("Stage definition must include e=<epoch>")
    return out

def _apply_actions_to_model(model, actions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Áp dụng 1 set hành động vào model. Trả về summary nhỏ cho logging.
    """
    summary = {}
    # 0) Unfreeze all (nếu yêu cầu)
    if actions.get("unfreeze_all", False):
        unfreeze_all(model)
        summary["unfreeze_all"] = True

    # 1) Encoder scopes
    if "enc_bottom_k" in actions:
        k = int(actions["enc_bottom_k"])
        summary["enc_bottom_k"] = freeze_bottom_k_layers(model, k)
    if "enc_bottom_ratio" in actions and "enc_bottom_k" not in actions:
        r = float(actions["enc_bottom_ratio"])
        summary["enc_bottom_ratio"] = freeze_bottom_ratio(model, r)
    if "enc_top_k" in actions:
        k = int(actions["enc_top_k"])
        summary["enc_top_k"] = freeze_top_k_layers(model, k)

    # 2) Blocks / Components
    if actions.get("pre", False):
        summary["pre"] = freeze_preprocessor(model)
    if actions.get("subs", False):
        summary["subs"] = freeze_subsampling(model)
    if actions.get("pos", False):
        summary["pos"] = freeze_pos_enc(model)
    if actions.get("spec", False):
        summary["spec"] = freeze_spec_augment(model)

    # 3) Decoder & Joint
    if actions.get("dec_all", False):
        summary["dec_all"] = freeze_decoder(model)
    else:
        if actions.get("dec_embed", False):
            summary["dec_embed"] = freeze_decoder_embedding(model)
        if actions.get("dec_rnn", False):
            summary["dec_rnn"] = freeze_decoder_rnn(model)
    if actions.get("joint", False):
        summary["joint"] = freeze_joint(model)

    # 4) Advanced
    if actions.get("prefix"):
        affected = freeze_by_prefixes(model, actions["prefix"])
        summary["prefix"] = {k: v for k, v in affected.items() if v > 0}
    if actions.get("regex"):
        rx_sum = []
        for pat in actions["regex"]:
            n = freeze_by_regex(model, pat)
            if n > 0:
                rx_sum.append((pat, n))
        if rx_sum:
            summary["regex"] = rx_sum
    if actions.get("types"):
        n = freeze_by_types(model, tuple(actions["types"]))
        if n > 0:
            summary["types"] = (actions["types"], n)

    # 5) Final counts
    trainable, total = count_params(model)
    summary["trainable"] = trainable
    summary["total"] = total
    summary["sample_frozen"] = snapshot_frozen(model, topk=10)
    return summary

class FreezeScheduleCallback(Callback):
    """
    Áp dụng nhiều stage freeze/unfreeze trong 1 lần train.
    - Mỗi stage có e=<epoch bắt đầu>.
    - Stage được áp dụng duy nhất 1 lần khi epoch hiện tại >= e và stage chưa applied.
    - Các stage sau có thể “ghi đè” logic đóng băng trước đó (vd unfreeze_all=1).
    """
    def __init__(self, stages: List[Dict[str, Any]], dump: bool = False):
        super().__init__()
        # Sort theo epoch
        self.stages = sorted(stages, key=lambda d: d["e"])
        self.applied = [False] * len(self.stages)
        self.dump = dump

    def on_train_start(self, trainer, pl_module):
        # Áp dụng những stage có e == 0 ngay lúc start
        self._maybe_apply_stages(trainer, pl_module, current_epoch=0, hook="on_train_start")

    def on_train_epoch_start(self, trainer, pl_module):
        cur = trainer.current_epoch or 0
        self._maybe_apply_stages(trainer, pl_module, current_epoch=cur, hook="on_train_epoch_start")

    def _maybe_apply_stages(self, trainer, pl_module, current_epoch: int, hook: str):
        for idx, st in enumerate(self.stages):
            if not self.applied[idx] and current_epoch >= st["e"]:
                print(f"❄️  [FreezeSchedule] Apply stage#{idx} at epoch {current_epoch}: {st}")
                summary = _apply_actions_to_model(pl_module, st)
                # Log ngắn gọn
                btm = summary.get("enc_bottom_k") or summary.get("enc_bottom_ratio")
                top = summary.get("enc_top_k")
                if btm:
                    print(f"   ↳ enc_bottom: {btm}")
                if top:
                    print(f"   ↳ enc_top_k: {top}")
                # dump (tùy chọn)
                if self.dump:
                    print(f"   ↳ Trainable params: {summary['trainable']:,}/{summary['total']:,}")
                    print(f"   ↳ Frozen sample: {summary.get('sample_frozen', [])}")
                self.applied[idx] = True

# -----------------------------
# Core helpers
# -----------------------------
def _set_requires_grad(m: nn.Module, flag: bool) -> int:
    """Set requires_grad cho toàn bộ tham số trong module m. Trả về số params bị ảnh hưởng."""
    n = 0
    for p in m.parameters(recurse=True):
        if p.requires_grad != flag:
            p.requires_grad = flag
        n += p.numel()
    return n

def _get_encoder_layers(model) -> List[nn.Module]:
    """Trả về list các layer Encoder (bottom->top) cho FastConformer NeMo."""
    enc = getattr(model, "encoder", None)
    if enc is None:
        return []
    layers = getattr(enc, "layers", None)
    if layers is None:
        return []
    return list(layers)

def count_params(model) -> Tuple[int, int]:
    """(trainable, total)"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

def snapshot_frozen(model, topk: int = 30) -> List[str]:
    """Liệt kê nhanh một số tham số đang freeze (để sanity-check)."""
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            out.append(name)
            if len(out) >= topk:
                break
    return out

# -----------------------------
# Existing layer granularity
# -----------------------------
def freeze_bottom_k_layers(model, k: int) -> int:
    layers = _get_encoder_layers(model)
    if k <= 0 or not layers:
        return 0
    k = min(k, len(layers))
    for i in range(k):
        _set_requires_grad(layers[i], False)
    return k

def freeze_top_k_layers(model, k: int) -> int:
    """Đóng băng k lớp trên cùng của encoder."""
    layers = _get_encoder_layers(model)
    if k <= 0 or not layers:
        return 0
    k = min(k, len(layers))
    for i in range(len(layers) - k, len(layers)):
        _set_requires_grad(layers[i], False)
    return k

def freeze_bottom_ratio(model, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    layers = _get_encoder_layers(model)
    if not layers:
        return 0
    k = max(1, int(round(len(layers) * ratio)))
    return freeze_bottom_k_layers(model, k)

def unfreeze_all(model):
    _set_requires_grad(model, True)

# -----------------------------
# Named components (by path)
# -----------------------------
def _get_by_path(model, path: str) -> Optional[nn.Module]:
    """Lấy module theo chuỗi path dạng 'encoder.pre_encode.conv' ..."""
    cur = model
    for seg in path.split("."):
        if not hasattr(cur, seg):
            return None
        cur = getattr(cur, seg)
    return cur

def freeze_by_prefixes(model, prefixes: Iterable[str]) -> Dict[str, int]:
    """
    Đóng băng theo prefix đường dẫn tham số (name trong named_parameters).
    Ví dụ: ["preprocessor", "encoder.pre_encode", "decoder", "joint"].
    """
    prefixes = list(prefixes)
    affected = {p: 0 for p in prefixes}
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in prefixes):
            p.requires_grad = False
            affected[next(pref for pref in prefixes if name.startswith(pref))] += p.numel()
    return affected

def freeze_modules(model, module_paths: Iterable[str]) -> Dict[str, int]:
    """
    Đóng băng theo tên module (object), an toàn hơn so với prefix.
    Ví dụ: ["preprocessor", "encoder.pre_encode", "encoder.pos_enc",
            "decoder", "joint", "spec_augmentation"]
    """
    res = {}
    for path in module_paths:
        mod = _get_by_path(model, path)
        if mod is not None:
            res[path] = _set_requires_grad(mod, False)
        else:
            res[path] = 0
    return res

# -----------------------------
# Common blocks (FastConformer NeMo)
# -----------------------------
def freeze_preprocessor(model) -> int:
    mod = getattr(model, "preprocessor", None)
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_subsampling(model) -> int:
    """encoder.pre_encode (ConvSubsampling + projection)."""
    mod = _get_by_path(model, "encoder.pre_encode")
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_pos_enc(model) -> int:
    mod = _get_by_path(model, "encoder.pos_enc")
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_spec_augment(model) -> int:
    """Tắt học tham số (thường SpecAugment không có params, nhưng ta để thống nhất)."""
    mod = _get_by_path(model, "spec_augmentation")
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_decoder(model) -> int:
    mod = getattr(model, "decoder", None)
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_decoder_embedding(model) -> int:
    mod = _get_by_path(model, "decoder.prediction.embed")
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_decoder_rnn(model) -> int:
    mod = _get_by_path(model, "decoder.prediction.dec_rnn")
    return _set_requires_grad(mod, False) if mod is not None else 0

def freeze_joint(model) -> int:
    mod = getattr(model, "joint", None)
    return _set_requires_grad(mod, False) if mod is not None else 0

# -----------------------------
# By regex on parameter names
# -----------------------------
def freeze_by_regex(model, pattern: str) -> int:
    """
    Đóng băng mọi tham số có tên khớp regex (trên named_parameters()).
    Ví dụ: r'^encoder\\.layers\\.(?:0|1|2)\\.' để khóa 3 lớp đầu.
    """
    rx = re.compile(pattern)
    n = 0
    for name, p in model.named_parameters():
        if rx.search(name):
            p.requires_grad = False
            n += p.numel()
    return n

# -----------------------------
# By module types
# -----------------------------
def freeze_by_types(model, types: Union[type, Tuple[type, ...]]) -> int:
    """
    Đóng băng theo kiểu lớp (vd LayerNorm, BatchNorm).
    Ví dụ: freeze_by_types(model, nn.LayerNorm)
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, types):
            n += _set_requires_grad(mod, False)
    return n

# -----------------------------
# High-level: config-driven
# -----------------------------
def apply_freeze_plan(model, plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    plan: dict cấu hình đông băng linh hoạt. Ví dụ:
    plan = {
        "encoder": {
            "freeze_bottom_k": 14,     # hoặc "freeze_bottom_ratio": 0.5
            "freeze_top_k": 0
        },
        "blocks": ["preprocessor", "subsampling", "pos_enc"],
        "decoder": {"freeze_all": True, "freeze_embed": False, "freeze_rnn": False},
        "joint": True,
        "specaug": True,
        "extra_prefixes": ["encoder.pre_encode.out"],   # tùy chọn
        "regex": [r"^encoder\.layers\.0\.", r"^encoder\.layers\.1\."],
        "freeze_types": ["LayerNorm"],  # tên lớp trong torch.nn cần freeze
    }
    """
    summary: Dict[str, Any] = {}

    # Encoder layers
    enc_plan = plan.get("encoder", {})
    b_k = enc_plan.get("freeze_bottom_k", 0)
    b_r = enc_plan.get("freeze_bottom_ratio", 0.0)
    t_k = enc_plan.get("freeze_top_k", 0)

    if b_k:
        summary["freeze_bottom_k_layers"] = freeze_bottom_k_layers(model, int(b_k))
    if b_r and not b_k:
        summary["freeze_bottom_ratio_layers"] = freeze_bottom_ratio(model, float(b_r))
    if t_k:
        summary["freeze_top_k_layers"] = freeze_top_k_layers(model, int(t_k))

    # Named blocks
    blocks = set(plan.get("blocks", []))
    if "preprocessor" in blocks:
        summary["preprocessor"] = freeze_preprocessor(model)
    if "subsampling" in blocks:
        summary["subsampling"] = freeze_subsampling(model)
    if "pos_enc" in blocks:
        summary["pos_enc"] = freeze_pos_enc(model)

    # Decoder
    dec_plan = plan.get("decoder", {})
    if dec_plan.get("freeze_all"):
        summary["decoder_all"] = freeze_decoder(model)
    else:
        if dec_plan.get("freeze_embed"):
            summary["decoder_embed"] = freeze_decoder_embedding(model)
        if dec_plan.get("freeze_rnn"):
            summary["decoder_rnn"] = freeze_decoder_rnn(model)

    # Joint
    if plan.get("joint"):
        summary["joint"] = freeze_joint(model)

    # SpecAug
    if plan.get("specaug"):
        summary["specaug"] = freeze_spec_augment(model)

    # Extra prefixes
    extra_pref = plan.get("extra_prefixes", [])
    if extra_pref:
        summary["extra_prefixes"] = freeze_by_prefixes(model, extra_pref)

    # Regex
    regex_list = plan.get("regex", [])
    regex_affected = []
    for pat in regex_list:
        regex_affected.append((pat, freeze_by_regex(model, pat)))
    if regex_affected:
        summary["regex"] = regex_affected

    # Types
    # Cho phép tên lớp phổ biến trong torch.nn
    freeze_types_names = plan.get("freeze_types", [])
    name2type = {
        "LayerNorm": nn.LayerNorm,
        "BatchNorm1d": nn.BatchNorm1d,
        "BatchNorm2d": nn.BatchNorm2d,
        "BatchNorm3d": nn.BatchNorm3d,
        "InstanceNorm1d": nn.InstanceNorm1d,
        "InstanceNorm2d": nn.InstanceNorm2d,
        "InstanceNorm3d": nn.InstanceNorm3d,
    }
    types_to_freeze = tuple(name2type[n] for n in freeze_types_names if n in name2type)
    if types_to_freeze:
        summary["types"] = freeze_by_types(model, types_to_freeze)

    # Final counts
    trainable, total = count_params(model)
    summary["trainable_params"] = trainable
    summary["total_params"] = total
    summary["sample_frozen_names"] = snapshot_frozen(model, topk=20)
    return summary

# -----------------------------
# Convenience presets
# -----------------------------
def preset_encoder_bottom14_decoder_joint(model) -> Dict[str, Any]:
    """
    Freeze giống log bạn dán: đóng băng 14 layer encoder dưới + decoder + joint + tiền xử lý,
    giữ train chủ yếu phần encoder top + joint head tùy chọn.
    """
    plan = {
        "encoder": {"freeze_bottom_k": 14},
        "blocks": ["preprocessor", "subsampling", "pos_enc"],  # tùy dự án bật/tắt
        "decoder": {"freeze_all": True},
        "joint": True,
        "specaug": False,
    }
    return apply_freeze_plan(model, plan)
