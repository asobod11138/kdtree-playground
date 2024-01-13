from __future__ import annotations

import random

import numpy as np
import pytest  # noqa: F401
from tqdm import tqdm

from kdtree_lib import KDTreeBasic, KDTreeOpen3d

points_num = 50_000
query_num = 200

def get_random_points(num_points) -> np.ndarray:
    return np.array([(random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(num_points)])  # noqa: S311


def build_tree_speed(arg_class, points) -> None:
    arg_class(points)


def kdtree_speed(tree, querys) -> None:
    for query in querys:
        tree.nearest_neighbor(query)

##########################################

def test_build_tree_kdtree_basic(benchmark) -> None:
    benchmark.group = "Tree Building"
    # テスト用のデータを準備
    points = get_random_points(points_num)

    benchmark(build_tree_speed, arg_class=KDTreeBasic, points=points)


def test_build_tree_kdtree_open3d(benchmark) -> None:
    benchmark.group = "Tree Building"
    # テスト用のデータを準備
    points = get_random_points(points_num)

    benchmark(build_tree_speed, arg_class=KDTreeOpen3d, points=points)


##########################################

def test_nearest_neighbor_kdtree_basic(benchmark) -> None:
    benchmark.group = "Nearest Neighbor Search"
    # テスト用のデータを準備
    points = get_random_points(points_num)
    tree = KDTreeBasic(points)

    querys = get_random_points(query_num)

    # nearest_neighborメソッドの実行速度を計測
    benchmark(kdtree_speed, tree=tree, querys=querys)



def test_nearest_neighbor_kdtree_open3d(benchmark) -> None:
    benchmark.group = "Nearest Neighbor Search"
    # テスト用のデータを準備
    points = get_random_points(points_num)
    tree = KDTreeOpen3d(points)

    querys = get_random_points(query_num)

    # nearest_neighborメソッドの実行速度を計測
    benchmark(kdtree_speed, tree=tree, querys=querys)
