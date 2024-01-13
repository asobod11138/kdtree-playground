from __future__ import annotations

import random

import numpy as np
import pytest  # noqa: F401
from tqdm import tqdm

from kdtree_lib import KDTreeBasic, KDTreeOpen3d


def get_random_points(num_points) -> np.ndarray:
    return np.array([(random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(num_points)])  # noqa: S311


def build_tree_speed(arg_class, points) -> None:
    arg_class(points)


def kdtree_speed(arg_class, tree, querys) -> None:
    for query in tqdm(querys, desc=f"kdtree speed: {arg_class.__name__}"):
        tree.nearest_neighbor(query)


def test_build_tree_kdtree_basic(benchmark) -> None:
    benchmark.group = "Tree Building"
    # テスト用のデータを準備
    points = get_random_points(1_000)

    # build_treeメソッドの実行速度を計測
    ret = benchmark.pedantic(
        build_tree_speed,
        kwargs={"arg_class": KDTreeBasic, "points": points},  # テスト対象に渡す引数 (キーワード付き)
        rounds=100,  # テスト対象の呼び出し回数
        iterations=10,
    )  # 試行回数


def test_nearest_neighbor_kdtree_basic(benchmark) -> None:
    benchmark.group = "Nearest Neighbor Search"
    # テスト用のデータを準備
    points = get_random_points(10_000)
    tree = KDTreeBasic(points)

    querys = get_random_points(200)

    # nearest_neighborメソッドの実行速度を計測
    benchmark(kdtree_speed, arg_class=KDTreeBasic, tree=tree, querys=querys)


def test_build_tree_kdtree_open3d(benchmark) -> None:
    benchmark.group = "Tree Building"
    # テスト用のデータを準備
    points = get_random_points(1_000)

    # build_treeメソッドの実行速度を計測
    ret = benchmark.pedantic(
        build_tree_speed,
        kwargs={"arg_class": KDTreeOpen3d, "points": points},  # テスト対象に渡す引数 (キーワード付き)
        rounds=100,  # テスト対象の呼び出し回数
        iterations=10,
    )  # 試行回数


def test_nearest_neighbor_kdtree_open3d(benchmark) -> None:
    benchmark.group = "Nearest Neighbor Search"
    # テスト用のデータを準備
    points = get_random_points(10_000)
    tree = KDTreeBasic(points)

    querys = get_random_points(200)

    # nearest_neighborメソッドの実行速度を計測
    benchmark(kdtree_speed, arg_class=KDTreeOpen3d, tree=tree, querys=querys)
