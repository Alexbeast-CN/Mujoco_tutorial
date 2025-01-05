import pytest
import numpy as np
from utils.geoUtils import (
    RDPSimplifier,
    normal_point_to_plane_distance,
    std_point_to_plane_distance,
)


class TestRDPSimplifier:
    def test_point_to_line_distance(self):
        # 测试点到线段的距离计算
        point = np.array([0, 1])
        line_start = np.array([0, 0])
        line_end = np.array([1, 0])

        distance = RDPSimplifier.point_to_line_distance(point, line_start, line_end)
        assert np.isclose(distance, 1.0)

        # 测试点在线段端点外的情况
        point = np.array([2, 0])
        distance = RDPSimplifier.point_to_line_distance(point, line_start, line_end)
        assert np.isclose(distance, 1.0)

        # 测试线段起点和终点相同的情况
        distance = RDPSimplifier.point_to_line_distance(point, line_start, line_start)
        assert np.isclose(distance, 2.0)

    def test_estimate_local_curvature(self):
        points = np.array([[0, 0], [1, 0], [1, 1]])

        # 测试90度转角的曲率
        curvature = RDPSimplifier.estimate_local_curvature(points, 1)
        assert np.isclose(curvature, np.pi / 2)

        # 测试边界点的曲率
        curvature = RDPSimplifier.estimate_local_curvature(points, 0)
        assert np.isclose(curvature, 0.0)

        curvature = RDPSimplifier.estimate_local_curvature(points, 2)
        assert np.isclose(curvature, 0.0)

    def test_simplify(self):
        # 创建一个简单的测试轨迹
        points = np.array(
            [
                [0, 0],
                [0.1, 0.1],  # 这个点应该被移除
                [1, 1],
                [2, 2],
                [2.1, 2.1],  # 这个点应该被移除
                [3, 3],
            ]
        )

        # 使用较大的epsilon进行简化
        simplified = RDPSimplifier.simplify(points, epsilon=0.5)
        assert len(simplified) < len(points)
        assert np.array_equal(simplified[0], points[0])  # 保留起点
        assert np.array_equal(simplified[-1], points[-1])  # 保留终点

    def test_remove_close_points(self):
        points = np.array(
            [
                [0, 0],
                [0.1, 0],  # 这个点应该被移除
                [1, 0],
                [1.05, 0],  # 这个点应该被移除
                [2, 0],
            ]
        )

        filtered = RDPSimplifier.remove_close_points(points, min_distance=0.2)
        assert len(filtered) == 3
        assert np.array_equal(filtered[0], points[0])
        assert np.array_equal(filtered[-1], points[-1])


def test_normal_point_to_plane_distance():
    # 测试点到平面的距离计算(点法式)
    point = np.array([1, 0, 0])
    plane_point = np.array([0, 0, 0])
    normal = np.array([1, 0, 0])  # x=0平面

    distance = normal_point_to_plane_distance(point, plane_point, normal)
    assert np.isclose(distance, 1.0)

    # 测试点在平面上的情况
    point = np.array([0, 1, 0])
    distance = normal_point_to_plane_distance(point, plane_point, normal)
    assert np.isclose(distance, 0.0)

    # 测试非单位法向量的情况
    normal = np.array([2, 0, 0])
    distance = normal_point_to_plane_distance(point, plane_point, normal)
    assert np.isclose(distance, 0.0)


def test_std_point_to_plane_distance():
    # 测试点到平面的距离计算(标准式)
    # 平面 x = 1 (x - 1 = 0)
    point = np.array([2, 0, 0])
    distance = std_point_to_plane_distance(point, 1, 0, 0, -1)
    assert np.isclose(distance, 1.0)

    # 测试点在平面上的情况
    point = np.array([1, 0, 0])
    distance = std_point_to_plane_distance(point, 1, 0, 0, -1)
    assert np.isclose(distance, 0.0)

    # 测试一般平面的情况 (x + y + z = 0)
    point = np.array([1, 1, 1])
    distance = std_point_to_plane_distance(point, 1, 1, 1, 0)
    expected_distance = 3 / np.sqrt(3)  # |1 + 1 + 1| / sqrt(1² + 1² + 1²)
    assert np.isclose(distance, expected_distance)
