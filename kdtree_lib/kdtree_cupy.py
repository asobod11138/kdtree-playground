# 最もシンプルなKDTreeの実装。numpyのみを使用している。アルゴリズムの理解とCPU実行でのベースライン
from __future__ import annotations

import cupy as cp
import numpy as np

from kdtree_lib.interface import KDTree, Node


class NodeCupy(Node):
    """KDTreeCupyのノード。"""

    def __init__(self, point: cp.ndarray, left: Node | None = None, right: Node | None = None) -> None:
        if not isinstance(point, cp.ndarray):
            raise TypeError(f"pointはCuPy配列である必要があります、{type(point)}が渡されました。")

        self.point: cp.ndarray = point
        self.left: Node | None = left
        self.right: Node | None = right


class KDTreeCupy(KDTree):
    """KDTreeCupyの実装"""

    def __init__(self, points: np.ndarray) -> None:
        if not isinstance(points, np.ndarray):
            raise TypeError(f"pointはNumPy配列である必要がありますが、{type(points)}が渡されました。")
        self.root: NodeCupy | None = self.build_tree(points)

    def build_tree(self, points: np.ndarray, depth: int = 0) -> NodeCupy | None:
        """KDTreeを構築する"""
        points_cp = cp.array(points)
        node = self._build_tree_imple(points_cp, depth)
        return NodeCupy.from_node(node)

    def _build_tree_imple(self, points: cp.ndarray, depth: int = 0) -> Node | None:
        """再帰的にKDTreeを構築する。必要に応じて各KDTreeの実装で拡張する。"""
        if len(points) == 0:
            return None

        if not isinstance(points, cp.ndarray):
            raise TypeError(f"pointはCupy配列である必要がありますが、{type(points)}が渡されました。")

        # 軸を選択（3次元の場合、x軸、y軸、z軸を交互に選択）
        k = len(points[0])  # ポイントの次元
        axis = depth % k

        # 中央値で分割
        sorted_points = points[points[:, axis].argsort()]
        median = len(points) // 2

        # 再帰的にサブツリーを構築
        return NodeCupy(
            point=sorted_points[median],
            left=self.build_tree(sorted_points[:median], depth + 1),
            right=self.build_tree(sorted_points[median + 1 :], depth + 1),
        )

    def _nearest_neighbor_impl(self, point: np.ndarray, depth: int = 0) -> tuple[float, np.ndarray | None]:
        """最近傍点を探索する"""
        if not isinstance(point, np.ndarray):
            raise TypeError(f"pointはNumPy配列である必要がありますが、{type(point)}が渡されました。")
        cp_point = cp.array(point)
        dist, nearest_point_cp = self._nearest(self.root, cp_point, depth)
        nearest_point = None if nearest_point_cp is None else cp.asnumpy(nearest_point_cp)
        return dist, nearest_point

    def _nearest(self, node: Node | None, point: np.ndarray, depth: int) -> tuple[float, cp.ndarray | None]:
        if not isinstance(point, cp.ndarray):
            raise TypeError(f"_nearestの入力のpointはCuPy配列である必要がありますが、{type(point)}が渡されました。")

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
        current_dist = float(cp.linalg.norm(cp.array(point) - cp.array(node.point)))

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
    points = np.array([(9, 6, 3), (4, 7, 5), (7, 2, 9), (-1, 0, 5), (-5, -1.5, -9.54), (0, 0, 0), (2, 3, 1)])
    tree = KDTreeCupy(points)

    # クエリポイント（3次元）
    query_point = np.array((2, 1.5, 0.5))
    distance, nearest_point = tree.nearest_neighbor(query_point)
    print(f"Nearest to {query_point} is {nearest_point} with distance {distance}")
