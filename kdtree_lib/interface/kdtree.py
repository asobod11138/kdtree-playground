# KDTreeのインターフェース

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Node:
    """KDTreeのノード。必要に応じて各KDTreeの実装で拡張する。"""

    def __init__(self, point: np.ndarray, left: Node | None = None, right: Node | None = None) -> None:
        if not isinstance(point, np.ndarray):
            raise TypeError(f"pointはNumPy配列である必要があります、{type(point)}が渡されました。")

        self.point: np.ndarray = point
        self.left: Node | None = left
        self.right: Node | None = right

    @classmethod
    def from_node(cls, node: Node) -> Node | None:
        """別のノードからノードを作成する"""
        if node is None:
            return None
        else:
            return cls(point=node.point, left=node.left, right=node.right)


class KDTree(ABC):
    """KDTreeのインターフェース"""

    def __init__(self, points: np.ndarray) -> None:
        """コンストラクタ"""
        if not isinstance(points, np.ndarray):
            raise TypeError("pointsはNumPy配列である必要があります。")
        self.root: Node | None = self.build_tree(points)

    @abstractmethod
    def build_tree(self, points: np.ndarray, depth: int = 0) -> Any:
        """KDTreeを構築する"""

    def _build_tree_imple(self, points: np.ndarray, depth: int = 0) -> Node | None:
        """再帰的にKDTreeを構築する。必要に応じて各KDTreeの実装で拡張する。"""
        if len(points) == 0:
            return None

        if not isinstance(points, np.ndarray):
            raise TypeError(f"pointはNumPy配列である必要がありますが、{type(points)}が渡されました。")

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

    def nearest_neighbor(self, point: np.ndarray) -> tuple[float, np.ndarray | None]:
        """最近傍点を探索する"""
        if not isinstance(point, np.ndarray):
            raise TypeError(f"pointはNumPy配列である必要がありますが、{type(point)}が渡されました。")

        return self._nearest_neighbor_impl(point)

    @abstractmethod
    def _nearest_neighbor_impl(self, point: np.ndarray) -> tuple[float, np.ndarray | None]:
        """最近傍点を探索する実装"""
