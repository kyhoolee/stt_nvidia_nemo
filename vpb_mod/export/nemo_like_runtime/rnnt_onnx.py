# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import onnxruntime as ort

class RNNTModulesONNX:
    def __init__(self, enc_path: str, pred_path: str, joint_path: str, providers=None):
        providers = providers or ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(enc_path, providers=providers)
        self.pred = ort.InferenceSession(pred_path, providers=providers)
        self.joint = ort.InferenceSession(joint_path, providers=providers)

        self.enc_in  = [i.name for i in self.enc.get_inputs()]
        self.enc_out = [o.name for o in self.enc.get_outputs()]

        self.pred_in  = [i.name for i in self.pred.get_inputs()]     # 2 or 3
        self.pred_out = [o.name for o in self.pred.get_outputs()]    # expect 2

        self.joint_in  = [i.name for i in self.joint.get_inputs()]
        self.joint_out = [o.name for o in self.joint.get_outputs()]

        # predictor: hỗ trợ 2 hoặc 3 inputs
        if len(self.pred_in) == 3:
            self._pred_has_len = True
        elif len(self.pred_in) == 2:
            self._pred_has_len = False
        else:
            raise RuntimeError(f"Unexpected predictor inputs: {self.pred_in}")

    def encode(self, processed_signal: torch.Tensor, processed_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.enc.run(
            self.enc_out,
            {
                self.enc_in[0]: processed_signal.numpy().astype(np.float32),
                self.enc_in[1]: processed_len.numpy().astype(np.int64),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])

    def predictor(self, targets: torch.Tensor, target_length: torch.Tensor, states_hc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pred_has_len:
            feed = {
                self.pred_in[0]: targets.numpy().astype(np.int64),
                self.pred_in[1]: target_length.numpy().astype(np.int64),
                self.pred_in[2]: states_hc.numpy().astype(np.float32),
            }
        else:
            # predictor.onnx của bạn: inputs=['targets','states_hc']
            feed = {
                self.pred_in[0]: targets.numpy().astype(np.int64),
                self.pred_in[1]: states_hc.numpy().astype(np.float32),
            }
        out = self.pred.run(self.pred_out, feed)
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])

    def joint_logits(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        out = self.joint.run(
            self.joint_out,
            {
                self.joint_in[0]: enc.numpy().astype(np.float32),
                self.joint_in[1]: pred.numpy().astype(np.float32),
            },
        )
        return torch.from_numpy(out[0])

class GreedyRNNTDecoder:
    def __init__(self, modules: RNNTModulesONNX, blank_id: int, vocab_size: int, L: int, H: int):
        self.m = modules
        self.blank_id = int(blank_id)
        self.vocab_size = int(vocab_size)
        self.L = int(L)
        self.H = int(H)

    def _predictor_step(self, y_prev_b: torch.Tensor, states_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # y_prev_b: [1]; states_b: [2,L(=1),1,H]
        targets = y_prev_b.view(1, 1)                 # [1,1]
        lengths = torch.ones(1, dtype=torch.long)     # chỉ dùng nếu _pred_has_len=True
        pred_seq, next_states = self.m.predictor(targets, lengths, states_b)  # pred_seq [1,U,H] (U=1 ở bước này)
        # Lấy phần tử cuối theo U (ổn cho cả U=1 hoặc U>1)
        pred_step = pred_seq[:, -1, :]  # [1,H]
        return pred_step, next_states   # [1,H], [2,L,1,H]

    def decode_batch(self, enc: torch.Tensor, enc_len: torch.Tensor, max_u: int = 256) -> list[list[int]]:
        B, T, Denc = enc.shape
        hyps: list[list[int]] = [[] for _ in range(B)]
        # L=1 theo onnx (xem predictor states_hc shape), nhưng vẫn lấy từ config để tương thích
        states = torch.zeros(2, self.L, B, self.H, dtype=torch.float32)
        y = torch.full((B,), self.blank_id, dtype=torch.long)

        # warmup predictor
        preds = []
        next_states_all = []
        for b in range(B):
            p_b, s_b = self._predictor_step(y[b:b+1], states[:, :, b:b+1, :])
            preds.append(p_b)               # [1,H]
            next_states_all.append(s_b)     # [2,L,1,H]
        preds = torch.cat(preds, dim=0)      # [B,H]
        states = torch.cat(next_states_all, dim=2)  # [2,L,B,H]

        for b in range(B):
            t = 0
            u = 0
            while t < int(enc_len[b]) and u < max_u:
                enc_t = enc[b:b+1, t:t+1, :]          # [1,1,512]
                pred_u = preds[b:b+1, :].unsqueeze(1) # [1,1,640]
                logits = self.m.joint_logits(enc_t, pred_u)  # [1,1,1,V]
                k = int(torch.argmax(logits[0,0,0]))
                if k == self.blank_id:
                    t += 1
                else:
                    hyps[b].append(k)
                    yb = torch.tensor([k], dtype=torch.long)
                    p_b, s_b = self._predictor_step(yb, states[:, :, b:b+1, :])
                    preds[b:b+1, :] = p_b
                    states[:, :, b:b+1, :] = s_b
                    u += 1
        return hyps

    def decode_one_with_trace(self, enc_b: torch.Tensor, enc_len_b: int, max_u: int = 256):
        """
        Decode 1 sample với trace chi tiết từng bước:
          - returns: (ids: List[int], trace: List[dict])
        enc_b: [1,T,512], enc_len_b: int
        """
        assert enc_b.dim() == 3 and enc_b.shape[0] == 1
        H = self.H
        # state [2,L,1,H]
        states = torch.zeros(2, self.L, 1, H, dtype=torch.float32)
        # y_prev = blank
        y_prev = torch.tensor([self.blank_id], dtype=torch.long)
        # warmup predictor
        pred_prev, states = self._predictor_step(y_prev, states)  # pred_prev [1,H]
        ids = []
        tr = []
        t = 0
        u = 0
        T = int(enc_len_b)
        while t < T and u < max_u:
            enc_t = enc_b[:, t:t+1, :]          # [1,1,512]
            pred_u = pred_prev.unsqueeze(1)     # [1,1,H]
            logits = self.m.joint_logits(enc_t, pred_u)  # [1,1,1,V]
            logit_vec = logits[0, 0, 0]         # [V]
            k = int(torch.argmax(logit_vec))
            is_blank = (k == self.blank_id)
            tr.append({
                "t": t, "u": u, "chosen_id": k, "is_blank": bool(is_blank),
            })
            if is_blank:
                t += 1
            else:
                ids.append(k)
                # feedback to predictor
                y_prev = torch.tensor([k], dtype=torch.long)
                pred_prev, states = self._predictor_step(y_prev, states)  # update pred_prev & states
                u += 1
        return ids, tr

    def decode_batch(self, enc: torch.Tensor, enc_len: torch.Tensor, max_u: int = 256) -> list[list[int]]:
        #  (giữ nguyên như bạn đang dùng)
        B, T, Denc = enc.shape
        hyps: list[list[int]] = [[] for _ in range(B)]
        states = torch.zeros(2, self.L, B, self.H, dtype=torch.float32)
        y = torch.full((B,), self.blank_id, dtype=torch.long)

        preds = []
        next_states_all = []
        for b in range(B):
            p_b, s_b = self._predictor_step(y[b:b+1], states[:, :, b:b+1, :])
            preds.append(p_b)
            next_states_all.append(s_b)
        preds = torch.cat(preds, dim=0)
        states = torch.cat(next_states_all, dim=2)

        for b in range(B):
            t = 0
            u = 0
            while t < int(enc_len[b]) and u < max_u:
                enc_t = enc[b:b+1, t:t+1, :]
                pred_u = preds[b:b+1, :].unsqueeze(1)
                logits = self.m.joint_logits(enc_t, pred_u)
                k = int(torch.argmax(logits[0,0,0]))
                if k == self.blank_id:
                    t += 1
                else:
                    hyps[b].append(k)
                    yb = torch.tensor([k], dtype=torch.long)
                    p_b, s_b = self._predictor_step(yb, states[:, :, b:b+1, :])
                    preds[b:b+1, :] = p_b
                    states[:, :, b:b+1, :] = s_b
                    u += 1
        return hyps