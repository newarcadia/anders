"""Matrix arithmetic for native model and animation transforms."""

from __future__ import annotations
from typing import List
from ..errors import UgxError

__all__ = ('mat4_mul', 'mat3_mul', 'mat3_inverse', 'mat4_inverse', 'uax_matrix4_identity', 'uax_matrix3_identity')

def mat4_mul(left: List[List[float]], right: List[List[float]]) -> List[List[float]]:
    return [[sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)] for row in range(4)]


def mat3_mul(left: List[List[float]], right: List[List[float]]) -> List[List[float]]:
    return [[sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)] for row in range(3)]


def mat3_inverse(matrix: List[List[float]]) -> List[List[float]]:
    rows4 = [[float(matrix[row][column]) for column in range(3)] + [0.0] for row in range(3)]
    rows4.append([0.0, 0.0, 0.0, 1.0])
    return [row[:3] for row in mat4_inverse(rows4)[:3]]


def mat4_inverse(matrix: List[List[float]]) -> List[List[float]]:
    inverse = [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]
    work = [[float(matrix[row][column]) for column in range(4)] for row in range(4)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda row: abs(work[row][column]))
        if work[pivot][column] == 0.0:
            raise UgxError("Matrix is singular and cannot be inverted.")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            inverse[column], inverse[pivot] = inverse[pivot], inverse[column]
        scale = work[column][column]
        for x in range(4):
            work[column][x] /= scale
            inverse[column][x] /= scale
        for row in range(4):
            if row == column:
                continue
            factor = work[row][column]
            if factor == 0.0:
                continue
            for x in range(4):
                work[row][x] -= factor * work[column][x]
                inverse[row][x] -= factor * inverse[column][x]
    return inverse


def uax_matrix4_identity() -> List[List[float]]:
    return [[1.0 if row == column else 0.0 for column in range(4)] for row in range(4)]


def uax_matrix3_identity() -> List[List[float]]:
    return [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
