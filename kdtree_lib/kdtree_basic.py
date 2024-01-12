# 最もシンプルなKDTreeの実装。numpyのみを使用している。アルゴリズムの理解とCPU実行でのベースライン
from __future__ import annotations

import numpy as np


class Node:
    """KDTreeのノード"""

    def __init__(self, point: np.ndarray | list | tuple, left: Node | None = None, right: Node | None = None) -> None:
        # pointsがリストの場合はNumPy配列に変換
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        elif not isinstance(point, np.ndarray):
            raise TypeError("pointはリストまたはNumPy配列である必要があります。")
        self.point: np.ndarray = point
        self.left: Node | None = left
        self.right: Node | None = right


class KDTreeBasic:
    """KDTreeBasicの実装"""

    def __init__(self, points: np.ndarray | list | tuple) -> None:
        # pointsがリストの場合はNumPy配列に変換
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        elif not isinstance(points, np.ndarray):
            raise TypeError("pointはリストまたはNumPy配列である必要があります。")
        self.root: Node | None = self.build_tree(points)

    def build_tree(self, points: np.ndarray, depth: int = 0) -> Node | None:
        """再帰的にKDTreeを構築する"""
        if len(points) == 0:
            return None

        # pointsがリストの場合はNumPy配列に変換
        if isinstance(points, list):
            points = np.array(points)
        elif not isinstance(points, np.ndarray):
            raise TypeError("pointsはリストまたはNumPy配列である必要があります。")

        # 軸を選択（3次元の場合、x軸、y軸、z軸を交互に選択）
        k = len(points[0])  # ポイントの次元
        axis = depth % k

        # 中央値で分割
        sorted_points = points[points[:, axis].argsort()]
        median = len(points) // 2

        # 再帰的にサブツリーを構築
        return Node(
            point=sorted_points[median],
            left=self.build_tree(sorted_points[:median], depth + 1),
            right=self.build_tree(sorted_points[median + 1 :], depth + 1),
        )

    def nearest_neighbor(self, point: np.ndarray | list | tuple, depth: int = 0) -> tuple[float, np.ndarray | None]:
        """最近傍点を探索する"""
        if isinstance(point, (list, tuple)):
            point = np.array(point)
        elif not isinstance(point, np.ndarray):
            raise TypeError("pointはリストまたはNumPy配列である必要があります。")
        dist, nearest_point = self._nearest(self.root, point, depth)
        return dist, nearest_point

    def _nearest(self, node: Node | None, point: np.ndarray, depth: int) -> tuple[float, np.ndarray | None]:
        if node is None:
            return float("inf"), None

        k = len(point)
        axis = depth % k

        next_branch = None
        opposite_branch = None
        if point[axis] < node.point[axis]:
            next_branch = node.left
            opposite_branch = node.right
        else:
            next_branch = node.right
            opposite_branch = node.left

        best_dist, best_point = self._nearest(next_branch, point, depth + 1)
        current_dist = float(np.linalg.norm(np.array(point) - np.array(node.point)))

        if current_dist < best_dist:
            best_dist = current_dist
            best_point = node.point

        # 現在のノードと他の分岐の距離を確認
        if abs(point[axis] - node.point[axis]) < best_dist:
            dist, _point = self._nearest(opposite_branch, point, depth + 1)
            if dist < best_dist:
                best_dist = dist
                best_point = _point

        return best_dist, best_point


# 3次元データ点の例
points = [(2, 3, 1), (5, 4, 2), (9, 6, 3), (4, 7, 5), (8, 1, 8), (7, 2, 9), (-1, 0, 5), (-5, -1.5, -9.54), (100, 100, 100), (0, 0, 0)]
tree = KDTreeBasic(points)

# クエリポイント（3次元）
query_point = (2.5, 3.5, 1.5)
distance, nearest_point = tree.nearest_neighbor(query_point)
print(f"Nearest to {query_point} is {nearest_point} with distance {distance}")
