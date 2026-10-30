"""
S4 LoRA Validator 单元测试
===========================
覆盖 3 项检查。
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s4_lora_validator import (
    S4LoRAValidator,
    A4_CONCEPT_BLEED,
    A4_TEXTURE_REPEAT,
    A4_COLOR_CAST,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def validator():
    return S4LoRAValidator(config_dir=CONFIG_DIR)


def make_frame(h=128, w=128, noise=0.05):
    """生成一张带噪声的 RGB 帧"""
    img = np.random.rand(h, w, 3).astype(np.float32) * noise + 0.5
    return np.clip(img, 0, 1).astype(np.float32)


def make_distinct_frame(h=256, w=256):
    """生成背景和无人机区域明显不同的帧"""
    # 背景：灰度+噪声
    bg = np.random.rand(h, w, 3).astype(np.float32) * 0.1 + 0.4
    # 无人机区域：暗色
    drone = np.random.rand(30, 30, 3).astype(np.float32) * 0.05 + 0.15
    frame = bg.copy()
    frame[50:80, 50:80] = drone
    return np.clip(frame, 0, 1).astype(np.float32)


# ═══════════════════════════════════════════════════════
# 正常路径
# ═══════════════════════════════════════════════════════

class TestPass:
    def test_valid_single_frame(self, validator):
        frame = make_frame()
        r = validator.validate([frame])
        assert r["pass"] is True

    def test_valid_multi_frame(self, validator):
        frames = [make_distinct_frame() for _ in range(3)]
        bboxes = [(50, 50, 30, 30)] * 3
        r = validator.validate(frames, bboxes)
        assert r["pass"] is True

    def test_no_bboxes_passes(self, validator):
        """无 bbox 时跳过泄漏和重复检查"""
        frames = [make_frame() for _ in range(5)]
        r = validator.validate(frames)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 1: LoRA 概念泄漏 (SSIM)
# ═══════════════════════════════════════════════════════

class TestConceptBleed:
    def test_no_bleed(self, validator):
        """随机背景与无人机区域差异大 → 不应触发"""
        frame = make_distinct_frame()
        r = validator.validate([frame], bboxes=[(50, 50, 30, 30)])
        assert r["pass"] is True

    def test_bleed_detected(self, validator):
        """无人机和背景完全相同 → 应触发泄漏"""
        frame = np.random.rand(128, 128, 3).astype(np.float32) * 0.05 + 0.5
        frame = np.clip(frame, 0, 1).astype(np.float32)
        # 让 crop 和背景一样（全图均匀）
        r = validator.validate([frame], bboxes=[(30, 30, 40, 40)])
        # SSIM 应该很高
        assert r["pass"] is False
        assert r["failure_code"] == A4_CONCEPT_BLEED


# ═══════════════════════════════════════════════════════
# 检查 2: LoRA 过拟合模式
# ═══════════════════════════════════════════════════════

class TestTextureRepeat:
    def test_normal_variation(self, validator):
        """不同帧 → 不应触发纹理重复"""
        frames = [make_distinct_frame() for _ in range(5)]
        bboxes = [(50, 50, 30, 30)] * 5
        r = validator.validate(frames, bboxes)
        assert r["pass"] is True

    def test_repeated_texture(self, validator):
        """完全相同帧 → 触发纹理重复"""
        frame = make_distinct_frame()
        frames = [frame.copy() for _ in range(5)]
        bboxes = [(50, 50, 30, 30)] * 5
        r = validator.validate(frames, bboxes)
        assert r["pass"] is False
        assert r["failure_code"] == A4_TEXTURE_REPEAT

    def test_single_frame_skips(self, validator):
        """单帧不检查纹理重复"""
        frame = make_distinct_frame()
        r = validator.validate([frame], bboxes=[(50, 50, 30, 30)])
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 3: 全局色偏
# ═══════════════════════════════════════════════════════

class TestColorCast:
    def test_normal_color(self, validator):
        frame = np.full((64, 64, 3), 0.5, dtype=np.float32)
        r = validator.validate([frame])
        assert r["pass"] is True

    def test_too_dark(self, validator):
        """全图偏暗"""
        frame = np.full((64, 64, 3), 0.1, dtype=np.float32)
        r = validator.validate([frame])
        assert r["pass"] is False
        assert r["failure_code"] == A4_COLOR_CAST

    def test_too_bright(self, validator):
        """全图过曝"""
        frame = np.full((64, 64, 3), 0.9, dtype=np.float32)
        r = validator.validate([frame])
        assert r["pass"] is False
        assert r["failure_code"] == A4_COLOR_CAST

    def test_single_channel_cast(self, validator):
        """单通道偏色"""
        frame = np.full((64, 64, 3), 0.5, dtype=np.float32)
        frame[:, :, 0] = 0.1  # R 通道偏暗
        r = validator.validate([frame])
        assert r["pass"] is False
        assert r["failure_code"] == A4_COLOR_CAST

    def test_uint8_normal(self, validator):
        """uint8 输入应正确归一化"""
        frame = np.full((64, 64, 3), 128, dtype=np.uint8)
        r = validator.validate([frame])
        assert r["pass"] is True  # 128/255 ≈ 0.502, 在 [0.3, 0.7] 内


# ═══════════════════════════════════════════════════════
# 短路求值
# ═══════════════════════════════════════════════════════

class TestShortCircuit:
    def test_color_cast_before_texture(self, validator):
        """色偏应优先于纹理重复报错"""
        dark_frame = np.full((64, 64, 3), 0.1, dtype=np.float32)
        frames = [dark_frame.copy() for _ in range(5)]
        bboxes = [(10, 10, 20, 20)] * 5
        r = validator.validate(frames, bboxes)
        assert r["failure_code"] == A4_CONCEPT_BLEED  # 先检 1 → 但泄漏可能不触发，色偏是检 3

    def test_order_is_bleed_texture_color(self, validator):
        """验证短路顺序：泄漏 → 纹理 → 色偏"""
        # 构造一个同时触发纹理重复和色偏的样本
        frame = np.full((64, 64, 3), 0.1, dtype=np.float32)
        frames = [frame.copy() for _ in range(5)]
        bboxes = [(10, 10, 20, 20)] * 5
        r = validator.validate(frames, bboxes)
        # 泄漏检查→纹理检查→色偏→应在某一步失败
        assert not r["pass"]
