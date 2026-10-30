"""
V5 — FailureBuffer：结构化失败日志 + 归档
==========================================

生成循环只写不读，学习循环只读不写。
对齐设计文档 §6 / §9.1 接口定义。

用法：
  buf = FailureBuffer("failure_log.jsonl", threshold=50)
  buf.record(failure_result)          # 生成循环写
  buf.count()                         # 当前累积数
  buf.is_ready()                      # 是否达阈值
  buf.archive_crop(image, sample_id)  # 归档失败 crop
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


class FailureBuffer:
    """失败样本缓冲池。

    Parameters
    ----------
    path : str
        JSONL 日志文件路径。
    threshold : int
        触发微调的累积阈值（默认 50）。
    archive_dir : str
        归档失败图像的目录。
    """

    def __init__(
        self,
        path: str = "failure_log.jsonl",
        threshold: int = 50,
        archive_dir: Optional[str] = None,
    ):
        self.path = Path(path)
        self.threshold = threshold
        self.archive_dir = Path(archive_dir) if archive_dir else self.path.parent / "crops"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    # -- 写入（生成循环调用）----------------------------------------------

    def record(
        self,
        sample_id: str,
        stage: str,
        failure_code: str,
        scores: dict,
        input_spec: Optional[dict] = None,
        gen_params: Optional[dict] = None,
        image_paths: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """记录一条失败日志条目。

        Parameters
        ----------
        sample_id : str
            样本标识。
        stage : str
            首次失败的阶段 (S6/S7/S8/S9)。
        failure_code : str
            失败码，如 S6_RGB_LOW_QUALITY。
        scores : dict
            各阶段分数，如 {"S6": 0.3, "S7": null, ...}。
        input_spec : dict, optional
            输入场景参数（drone_type, weather, scene_type 等）。
        gen_params : dict, optional
            生成参数（seed, steps, lora_weight 等）。
        image_paths : dict, optional
            {"rgb": "...", "ir": "..."}。
        extra : dict, optional
            其他字段。
        """
        entry = {
            "sample_id": sample_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "failure_code": failure_code,
            "scores": scores,
        }
        if input_spec:
            entry["input_spec"] = input_spec
        if gen_params:
            entry["gen_params"] = gen_params
        if image_paths:
            entry["images"] = image_paths
        if extra:
            entry["extra"] = extra

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def archive_crop(self, image: np.ndarray, sample_id: str) -> str:
        """保存失败帧的 RGB 图像到归档目录。"""
        import cv2
        fname = f"{sample_id}.png"
        dst = self.archive_dir / fname
        cv2.imwrite(str(dst), image)
        return str(dst)

    # -- 读取（学习循环调用）----------------------------------------------

    def count(self) -> int:
        """返回当前失败日志条目数。"""
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def is_ready(self) -> bool:
        """是否达到微调触发阈值。"""
        return self.count() >= self.threshold

    def load_all(self) -> list[dict]:
        """加载全部失败日志条目。"""
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def stats(self) -> dict:
        """返回失败日志统计摘要。"""
        entries = self.load_all()
        if not entries:
            return {"total": 0, "by_stage": {}, "by_code": {}}

        by_stage = {}
        by_code = {}
        for e in entries:
            s = e.get("stage", "?")
            by_stage[s] = by_stage.get(s, 0) + 1
            c = e.get("failure_code", "?")
            by_code[c] = by_code.get(c, 0) + 1

        return {
            "total": len(entries),
            "by_stage": by_stage,
            "by_code": by_code,
            "threshold": self.threshold,
            "ready": self.is_ready(),
        }

    def clear(self) -> None:
        """清空失败日志（微调成功后调用）。"""
        if self.path.exists():
            self.path.unlink()
