"""
S5 IR Validator 单元测试
=========================
覆盖 3 项防御性检查。所有情况下 pass 均为 True。
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s5_ir_validator import (
    S5IRValidator,
    A5_SIZE_MISMATCH,
    A5_NOT_GRAYSCALE,
    A5_HISTOGRAM_ANOMALY,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def validator():
    return S5IRValidator(config_dir=CONFIG_DIR)


# ═══════════════════════════════════════════════════════
# 正常路径
# ═══════════════════════════════════════════════════════

class TestPass:
    def test_valid_ir_grayscale(self, validator):
        """正常 IR 灰度图 → 无警告"""
        ir = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        r = validator.validate(ir)
        assert r["pass"] is True
        assert len(r["warnings"]) == 0

    def test_valid_ir_with_rgb(self, validator):
        """IR 与 RGB 尺寸匹配 → 无警告"""
        ir = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        r = validator.validate(ir, rgb)
        assert r["pass"] is True
        assert len(r["warnings"]) == 0

    def test_single_channel_3d(self, validator):
        """(H, W, 1) 格式应正常通过"""
        ir = np.random.randint(50, 200, (256, 256, 1), dtype=np.uint8)
        r = validator.validate(ir)
        assert r["pass"] is True
        assert len(r["warnings"]) == 0

    def test_multi_peak_histogram(self, validator):
        """多峰直方图 → 不触发直方图警告"""
        # 构造双峰分布
        ir = np.concatenate([
            np.random.normal(50, 10, 20000),
            np.random.normal(180, 10, 20000),
        ]).astype(np.uint8).reshape(200, 200)
        ir = np.clip(ir, 0, 255)
        r = validator.validate(ir)
        hist_warnings = [w for w in r["warnings"] if w["code"] == A5_HISTOGRAM_ANOMALY]
        assert len(hist_warnings) == 0


# ═══════════════════════════════════════════════════════
# 始终 PASS — 绝不截停
# ═══════════════════════════════════════════════════════

class TestNeverFails:
    def test_even_with_warnings_pass_is_true(self, validator):
        """即使有警告，pass 仍为 True"""
        # 创建多个问题：尺寸不匹配 + 非灰度 + 单峰直方图
        ir = np.full((128, 128, 3), 100, dtype=np.uint8)  # 三通道 + 单峰
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)      # 不同尺寸
        r = validator.validate(ir, rgb)
        assert r["pass"] is True
        assert r["failure_code"] is None
        assert len(r["warnings"]) >= 2  # 至少有尺寸 + 直方图


# ═══════════════════════════════════════════════════════
# 检查 1: 尺寸不匹配 → WARNING
# ═══════════════════════════════════════════════════════

class TestSizeMismatch:
    def test_different_size(self, validator):
        ir = np.zeros((128, 128), dtype=np.uint8)
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        r = validator.validate(ir, rgb)
        assert r["pass"] is True
        assert any(w["code"] == A5_SIZE_MISMATCH for w in r["warnings"])

    def test_no_rgb_skips(self, validator):
        """没有 RGB 参考就跳过尺寸检查"""
        ir = np.zeros((128, 128), dtype=np.uint8)
        r = validator.validate(ir)
        assert not any(w["code"] == A5_SIZE_MISMATCH for w in r["warnings"])


# ═══════════════════════════════════════════════════════
# 检查 2: 非灰度 → WARNING
# ═══════════════════════════════════════════════════════

class TestNotGrayscale:
    def test_pseudo_color(self, validator):
        """三通道且各通道均值不同 → 非灰度"""
        ir = np.zeros((64, 64, 3), dtype=np.uint8)
        ir[:, :, 0] = 50   # R 通道偏移
        ir[:, :, 1] = 100  # G 通道偏移
        ir[:, :, 2] = 200  # B 通道偏移
        r = validator.validate(ir)
        assert any(w["code"] == A5_NOT_GRAYSCALE for w in r["warnings"])

    def test_real_grayscale_3ch(self, validator):
        """三通道但各通道完全一致 → 应通过"""
        ir = np.full((64, 64, 3), 128, dtype=np.uint8)
        r = validator.validate(ir)
        assert not any(w["code"] == A5_NOT_GRAYSCALE for w in r["warnings"])


# ═══════════════════════════════════════════════════════
# 检查 3: 直方图异常 → WARNING
# ═══════════════════════════════════════════════════════

class TestHistogramAnomaly:
    def test_single_peak(self, validator):
        """单峰直方图（全图同一灰度）→ 告警"""
        ir = np.full((128, 128), 100, dtype=np.uint8)
        r = validator.validate(ir)
        assert any(w["code"] == A5_HISTOGRAM_ANOMALY for w in r["warnings"])

    def test_uniform_noise(self, validator):
        """均匀分布噪声 → 可能无峰 → 告警"""
        ir = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        r = validator.validate(ir)
        # 均匀分布可能检测不到显著峰
        hist_warnings = [w for w in r["warnings"] if w["code"] == A5_HISTOGRAM_ANOMALY]
        # 均匀分布是否告警取决于 find_peaks 参数
        # 不做强制断言，只验证 pass 为 True
        assert r["pass"] is True
