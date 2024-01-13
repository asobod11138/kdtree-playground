from __future__ import annotations

import numpy as np
import pytest  # noqa: F401

from kdtree_lib import KDTreeBasic, KDTreeOpen3d


def nearest_neighbor(arg_class):
    points = np.array([(2, 3, 1), (5, 4, 2), (9, 6, 3), (4, 7, 5), (8, 1, 8), (7, 2, 9)])
    query_point = np.array((2.5, 3.5, 1.5))
    tree = arg_class(points)
    distance, nearest_point = tree.nearest_neighbor(query_point)
    assert np.array_equal(nearest_point, (2, 3, 1))
    assert np.isclose(distance, np.linalg.norm(np.array(query_point) - np.array((2, 3, 1))))


def dense_points(arg_class):
    points = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)])
    query = np.array((1, 1, 1))
    tree = arg_class(points)
    distance, nearest_point = tree.nearest_neighbor(query)
    assert np.array_equal(nearest_point, (1, 1, 1))
    assert distance == 0


def near_boundary(arg_class):
    points = np.array([(0, 0, 0), (10, 10, 10), (20, 20, 20)])
    query = np.array((9, 9, 9))
    tree = arg_class(points)
    distance, nearest_point = tree.nearest_neighbor(query)
    assert np.array_equal(nearest_point, (10, 10, 10))


def far_from_any_points(arg_class):
    points = np.array([(0, 0, 0), (10, 10, 10), (20, 20, 20)])
    query = np.array((100, 100, 100))
    tree = arg_class(points)
    distance, nearest_point = tree.nearest_neighbor(query)
    assert np.array_equal(nearest_point, (20, 20, 20))


def query_outside_points(arg_class):
    points = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    query = np.array((0, 0, 0))
    tree = arg_class(points)
    distance, nearest_point = tree.nearest_neighbor(query)
    assert np.array_equal(nearest_point, (1, 2, 3))


def sub(arg_class: [KDTreeBasic]):
    nearest_neighbor(arg_class)
    dense_points(arg_class)
    near_boundary(arg_class)
    far_from_any_points(arg_class)
    query_outside_points(arg_class)


def test_main():
    sub(KDTreeBasic)
    sub(KDTreeOpen3d)
