import numpy as np


class RDPSimplifier:
    """
    Ramer-Douglas-Peucker (RDP) 算法实现,用于简化点序列。
    """

    @staticmethod
    def point_to_line_distance(point, line_start, line_end):
        """
        计算点到线段的垂直距离。

        参数：
        - point: 待计算的点
        - line_start: 线段起点
        - line_end: 线段终点

        返回：
        - 点到线段的垂直距离
        """
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)

        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len

        t = np.dot(line_unitvec, point_vec_scaled)

        # 如果投影点在线段之外
        if t < 0.0:
            return np.linalg.norm(point_vec)
        elif t > 1.0:
            return np.linalg.norm(point - line_end)

        # 投影点在线段上,计算垂直距离
        proj = line_start + line_unitvec * t * line_len
        return np.linalg.norm(point - proj)

    @staticmethod
    def _rdp_reduce(points, start_index, end_index, epsilon, mask):
        """
        RDP算法的递归实现。

        参数：
        - points: 轨迹点数组
        - start_index: 当前段的起始索引
        - end_index: 当前段的结束索引
        - epsilon: 简化阈值(可以是标量或数组)
        - mask: 标记需要保留的点
        """
        if end_index <= start_index + 1:
            return

        dmax = 0.0
        index = start_index

        start_point = points[start_index]
        end_point = points[end_index]

        for i in range(start_index + 1, end_index):
            d = RDPSimplifier.point_to_line_distance(points[i], start_point, end_point)
            if d > dmax:
                index = i
                dmax = d

        # 使用当前位置的 epsilon 值
        eps_threshold = epsilon[index] if isinstance(epsilon, np.ndarray) else epsilon
        if dmax > eps_threshold:
            mask[index] = True
            RDPSimplifier._rdp_reduce(points, start_index, index, epsilon, mask)
            RDPSimplifier._rdp_reduce(points, index, end_index, epsilon, mask)

    @staticmethod
    def estimate_local_curvature(points, i):
        """估计局部曲率"""
        if i < 1 or i >= len(points) - 1:
            return 0.0

        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]

        # 计算夹角变化
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    @staticmethod
    def simplify_indices(points, epsilon):
        """返回简化后点的索引

        Args:
            points: 点序列
            epsilon: 简化阈值

        Returns:
            保留点的索引数组
        """
        if len(points) < 3:
            return np.arange(len(points))

        # 计算每个点的曲率
        curvatures = np.array(
            [
                RDPSimplifier.estimate_local_curvature(points, i)
                for i in range(len(points))
            ]
        )

        # 根据曲率调整epsilon
        max_curvature = np.max(curvatures)
        if max_curvature > 0:
            adaptive_epsilon = epsilon * (1 - 0.5 * curvatures / max_curvature)
        else:
            adaptive_epsilon = np.full_like(curvatures, epsilon)

        mask = np.zeros(len(points), dtype=bool)
        mask[0] = mask[-1] = True

        RDPSimplifier._rdp_reduce(points, 0, len(points) - 1, adaptive_epsilon, mask)

        return np.where(mask)[0]

    @staticmethod
    def simplify(points, epsilon):
        """使用考虑曲率的RDP算法简化点序列

        Args:
            points: 点序列
            epsilon: 简化阈值

        Returns:
            简化后的点序列
        """
        indices = RDPSimplifier.simplify_indices(points, epsilon)
        return points[indices]

    @staticmethod
    def remove_close_points_indices(points, min_distance):
        """
        返回距离大于阈值的点的索引。

        参数：
        - points: numpy数组，形状为(N, D)的点序列
        - min_distance: float，两点之间的最小距离阈值

        返回：
        - numpy数组，保留点的索引
        """
        if len(points) < 2:
            return np.arange(len(points))

        # 初始化mask，默认保留所有点
        mask = np.ones(len(points), dtype=bool)

        # 计算相邻点之间的距离
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)

        # 标记距离过近的点
        close_points = distances < min_distance

        # 在每对过近的点中，移除后一个点
        mask[1:][close_points] = False

        # 返回保留点的索引
        return np.where(mask)[0]

    @staticmethod
    def remove_close_points(points, min_distance):
        """
        删除序列中距离过近的连续点。

        参数：
        - points: numpy数组，形状为(N, D)的点序列
        - min_distance: float，两点之间的最小距离阈值

        返回：
        - numpy数组，过滤后的点序列
        """
        indices = RDPSimplifier.remove_close_points_indices(points, min_distance)
        return points[indices]


def normal_point_to_plane_distance(point, plane_point, normal):
    """
    计算点到平面的距离(使用点法式)

    参数:
    point: array-like, 待计算距离的点坐标 [x, y, z]
    plane_point: array-like, 平面上一点的坐标 [x, y, z]
    normal: array-like, 平面的法向量 [a, b, c]

    返回:
    float: 点到平面的最短距离
    """
    # 将输入转换为numpy数组
    point = np.array(point)
    plane_point = np.array(plane_point)
    normal = np.array(normal)

    # 归一化法向量
    normal = normal / np.linalg.norm(normal)

    # 计算向量 point - plane_point
    vector = point - plane_point

    # 点到平面的距离就是 vector 在法向量方向上的投影长度
    distance = abs(np.dot(vector, normal))

    return distance


def std_point_to_plane_distance(point, a, b, c, d):
    """
    计算点到平面的距离(使用平面标准式 ax + by + cz + d = 0)

    参数:
    point: array-like, 点的坐标 [x, y, z]
    a, b, c: float, 平面标准式中的系数
    d: float, 平面标准式中的常数项

    返回:
    float: 点到平面的最短距离
    """
    # 将点坐标转换为numpy数组
    point = np.array(point)
    # 构造法向量
    normal = np.array([a, b, c])

    # 点到平面的距离公式: |ax₀ + by₀ + cz₀ + d| / √(a² + b² + c²)
    numerator = abs(np.dot(normal, point) + d)
    denominator = np.linalg.norm(normal)

    return numerator / denominator
