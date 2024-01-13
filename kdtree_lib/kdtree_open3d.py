# open3dの実装を使用したkd-treeの実装
from __future__ import annotations

import numpy as np
import open3d as o3d

from kdtree_lib.interface import KDTree


class KDTreeOpen3d(KDTree):
    """Open3DのKDTreeの実装"""

    def __init__(self, points: np.ndarray) -> None:
        """コンストラクタ"""
        if not isinstance(points, np.ndarray):
            raise TypeError("pointsはNumPy配列である必要があります。")
        self.points: np.ndarray = points
        self.tree: o3d.geometry.KDTreeFlann = self.build_tree(self.points)

    def build_tree(self, points: np.ndarray) -> o3d.geometry.KDTreeFlann:
        """KDTreeを構築する"""
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        # self.pcdではなく、ローカル変数としてpcdを定義するとガベージコレクションで解放されてしまい（たぶん）、近傍探索の結果がおかしくなる
        return o3d.geometry.KDTreeFlann(self.pcd)

    def _nearest_neighbor_impl(self, point: np.ndarray | list | tuple) -> tuple[float, np.ndarray | None]:
        """最近傍点を探索する"""
        [k, idx, _] = self.tree.search_knn_vector_3d(point, 1)
        print(k, idx, _)
        nearest_point = self.points[idx[0]]
        dist = float(np.linalg.norm(np.array(point) - nearest_point))
        return dist, nearest_point


if __name__ == "__main__":
    # 3次元データ点の例
    points = np.array([(2, 3, 1), (5, 4, 2), (9, 6, 3), (4, 7, 5), (8, 1, 8), (7, 2, 9)])
    tree = KDTreeOpen3d(points)

    # クエリポイント（3次元）
    query_point = np.array((2.5, 3.5, 1.5))
    distance, nearest_point = tree.nearest_neighbor(query_point)
    print(f"KDTree data: {tree.tree}")
    print(f"Nearest to {query_point} is {nearest_point} with distance {distance}")
