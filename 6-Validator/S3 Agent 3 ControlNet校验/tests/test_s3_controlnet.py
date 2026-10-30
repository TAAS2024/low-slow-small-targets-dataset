"""
S3 ControlNet Validator 单元测试
=================================
覆盖 4 项检查。
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from s3_controlnet_validator import (
    S3ControlNetValidator,
    A3_SEG_POSITION_OFFSET,
    A3_SEG_BOUNDARY_ARTIFACT,
    A3_DEPTH_MISALIGN,
    A3_MAP_SIZE_MISMATCH,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


@pytest.fixture
def validator():
    return S3ControlNetValidator(config_dir=CONFIG_DIR)


def make_maps(h=512, w=512):
    """生成合法的 seg/depth/pose 三件套"""
    seg = np.zeros((h, w), dtype=np.uint8)
    seg[200:230, 250:280] = 1  # 小无人机区域
    depth = np.random.rand(h, w).astype(np.float32) * 0.8 + 0.2
    # 让无人机区域 depth 略低（更近）
    depth[200:230, 250:280] = np.random.rand(30, 30).astype(np.float32) * 0.1
    pose = np.random.rand(h, w).astype(np.float32)
    return seg, depth, pose


# ═══════════════════════════════════════════════════════
# 正常路径
# ═══════════════════════════════════════════════════════

class TestPass:
    def test_valid_all(self, validator):
        seg, depth, pose = make_maps()
        r = validator.validate(seg, depth, pose, bbox_center=(265, 215))
        assert r["pass"] is True

    def test_no_bbox_passes(self, validator):
        """没有 bbox 时跳过位置检查"""
        seg, depth, pose = make_maps()
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is True


# ═══════════════════════════════════════════════════════
# 检查 1: 尺寸一致
# ═══════════════════════════════════════════════════════

class TestMapSize:
    def test_different_sizes(self, validator):
        seg = np.zeros((512, 512), dtype=np.uint8)
        depth = np.zeros((256, 256), dtype=np.float32)
        pose = np.zeros((512, 512), dtype=np.float32)
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is False
        assert r["failure_code"] == A3_MAP_SIZE_MISMATCH

    def test_same_sizes(self, validator):
        seg, depth, pose = make_maps()
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is True  # 不会触发尺寸错误


# ═══════════════════════════════════════════════════════
# 检查 2: seg 位置对齐
# ═══════════════════════════════════════════════════════

class TestSegPosition:
    def test_close_alignment(self, validator):
        seg, depth, pose = make_maps()
        # bbox 中心在无人机区域附近
        r = validator.validate(seg, depth, pose, bbox_center=(265, 215))
        assert r["pass"] is True

    def test_far_offset(self, validator):
        seg, depth, pose = make_maps()
        # bbox 中心远离无人机区域
        r = validator.validate(seg, depth, pose, bbox_center=(100, 100))
        assert r["pass"] is False
        assert r["failure_code"] == A3_SEG_POSITION_OFFSET

    def test_empty_seg_with_bbox(self, validator):
        """seg 全零但有 bbox → 应失败"""
        seg = np.zeros((512, 512), dtype=np.uint8)
        depth = np.random.rand(512, 512).astype(np.float32)
        pose = np.random.rand(512, 512).astype(np.float32)
        r = validator.validate(seg, depth, pose, bbox_center=(256, 256))
        assert r["pass"] is False
        assert r["failure_code"] == A3_SEG_POSITION_OFFSET


# ═══════════════════════════════════════════════════════
# 检查 3: seg 边界质量
# ═══════════════════════════════════════════════════════

class TestSegBoundary:
    def test_smooth_boundary(self, validator):
        seg, depth, pose = make_maps()
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is True

    def test_jagged_boundary(self, validator):
        """棋盘格锯齿 → 1px 膨胀填满所有空隙，增长率 ~100%"""
        seg = np.zeros((512, 512), dtype=np.uint8)
        for i in range(100, 200):
            for j in range(100, 200):
                if (i + j) % 2 == 0:
                    seg[i, j] = 1
        depth = np.random.rand(512, 512).astype(np.float32)
        pose = np.random.rand(512, 512).astype(np.float32)
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is False
        assert r["failure_code"] == A3_SEG_BOUNDARY_ARTIFACT

    def test_empty_seg_no_boundary_check(self, validator):
        """全零 seg 跳过边界检查"""
        seg = np.zeros((512, 512), dtype=np.uint8)
        depth = np.random.rand(512, 512).astype(np.float32)
        pose = np.random.rand(512, 512).astype(np.float32)
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is True  # 只检查尺寸 → 通过


# ═══════════════════════════════════════════════════════
# 检查 4: depth 一致性
# ═══════════════════════════════════════════════════════

class TestDepthConsistency:
    def test_valid_depth(self, validator):
        seg, depth, pose = make_maps()
        r = validator.validate(seg, depth, pose, scale_factor=0.5)
        assert r["pass"] is True

    def test_no_scale_factor_passes(self, validator):
        seg, depth, pose = make_maps()
        r = validator.validate(seg, depth, pose)
        assert r["pass"] is True

    def test_uniform_depth(self, validator):
        """全图 depth 一致 → 无人机与背景无差异"""
        seg, _, pose = make_maps()
        depth = np.full((512, 512), 0.5, dtype=np.float32)
        r = validator.validate(seg, depth, pose, scale_factor=0.5)
        # 由于无人机区域也是 0.5，差异为 0，应触发 A3_DEPTH_MISALIGN
        assert r["pass"] is False
        assert r["failure_code"] == A3_DEPTH_MISALIGN


# ═══════════════════════════════════════════════════════
# 短路求值
# ═══════════════════════════════════════════════════════

class TestShortCircuit:
    def test_size_before_position(self, validator):
        """尺寸不一致应先于位置对齐报错"""
        seg = np.zeros((512, 512), dtype=np.uint8)
        depth = np.zeros((256, 256), dtype=np.float32)
        pose = np.zeros((512, 512), dtype=np.float32)
        r = validator.validate(seg, depth, pose, bbox_center=(100, 100))
        assert r["failure_code"] == A3_MAP_SIZE_MISMATCH
