# 最もシンプルなKDTreeの実装。numpyのみを使用している。アルゴリズムの理解とCPU実行でのベースライン
from __future__ import annotations

import numpy as np

from kdtree_lib.interface import KDTree, Node


class NodeBasic(Node):
    """KDTreeBasicのノード"""

    # 特に変更や拡張が必要ないのでそのまま使用する


class KDTreeBasic(KDTree):
    """KDTreeBasicの実装"""

    def __init__(self, points: np.ndarray | list | tuple) -> None:
        # pointsがリストの場合はNumPy配列に変換
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        elif not isinstance(points, np.ndarray):
            raise TypeError("pointはリストまたはNumPy配列である必要があります。")
        self.root: NodeBasic | None = self.build_tree(points)

    def build_tree(self, points: np.ndarray, depth: int = 0) -> NodeBasic | None:
        """KDTreeを構築する"""
        node = self._build_tree_imple(points, depth)
        return NodeBasic.from_node(node)

    def _nearest_neighbor_impl(self, point: np.ndarray | list | tuple, depth: int = 0) -> tuple[float, np.ndarray | None]:
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


if __name__ == "__main__":
    # 3次元データ点の例
    points = np.array([(2, 3, 1), (9, 6, 3), (4, 7, 5), (7, 2, 9), (-1, 0, 5), (-5, -1.5, -9.54), (0, 0, 0)])
    tree = KDTreeBasic(points)

    # クエリポイント（3次元）
    query_point = np.array((2.5, 3.5, 1.5))
    distance, nearest_point = tree.nearest_neighbor(query_point)
    print(f"Nearest to {query_point} is {nearest_point} with distance {distance}")
