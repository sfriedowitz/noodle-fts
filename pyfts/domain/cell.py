from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

PI_HALF: float = np.pi / 2
PI_THIRD: float = np.pi / 3


def shape_tensor_1d(a: float) -> npt.NDArray:
    return np.array([[a]])


def shape_tensor_2d(a: float, b: float, gamma: float) -> npt.NDArray:
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    bx = b * cos_gamma
    by = b * sin_gamma

    return np.array(
        [
            [a, bx],
            [0.0, by],
        ]
    )


def shape_tensor_3d(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> npt.NDArray:
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.cos(gamma)

    ax = a
    bx = b * cos_gamma
    by = b * sin_gamma
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = (c * c - cx * cx - cy * cy).sqrt()

    return np.array(
        [
            [ax, bx, cx],
            [0.0, by, cy],
            [0.0, 0.0, cz],
        ]
    )


class UnitCell(ABC):
    """Base class for 1/2/3 dimensional unit cells.

    A `UnitCell` contains any cell parameters required to define the cell shape tensor.
    """

    @abstractmethod
    def shape_tensor(self) -> npt.NDArray:
        """Return the cell shape tensor derived from the cell parameters.

        The shape tensor in three dimensions is represented as `h = (a1, a2, a3)`
        where `a1`, `a2`, and `a3` are the three Bravais lattice vectors in columnar format.
        """
        raise NotImplementedError

    def metric_tensor(self) -> npt.NDArray:
        h = self.shape_tensor()
        return h.T @ h


class LamellarCell(UnitCell):
    def __init__(self, a: float):
        self.a = a

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_1d(self.a)


class SquareCell(UnitCell):
    def __init__(self, a: float):
        self.a = a

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_2d(self.a, self.a, PI_HALF)


class RectangularCell(UnitCell):
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_2d(self.a, self.b, PI_HALF)


class HexagonalCell(UnitCell):
    def __init__(self, a: float):
        self.a = a

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_2d(self.a, self.a, PI_THIRD)


class ObliqueCell(UnitCell):
    def __init__(self, a: float, b: float, gamma: float):
        self.a = a
        self.b = b
        self.gamma = gamma

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_2d(self.a, self.b, self.gamma)


class CubicCell(UnitCell):
    def __init__(self, a: float):
        self.a = a

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.a, self.a, PI_HALF, PI_HALF, PI_HALF)


class TetragonalCell(UnitCell):
    def __init__(self, a: float, c: float):
        self.a = a
        self.c = c

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.a, self.c, PI_HALF, PI_HALF, PI_HALF)


class OrthorhombicCell(UnitCell):
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.b, self.c, PI_HALF, PI_HALF, PI_HALF)


class TrigonalCell(UnitCell):
    def __init__(self, a: float, alpha: float):
        self.a = a
        self.alpha = alpha

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.a, self.a, self.alpha, self.alpha, self.alpha)


class HexagonalCell3D(UnitCell):
    def __init__(self, a: float, c: float):
        self.a = a
        self.c = c

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.a, self.c, PI_HALF, PI_HALF, PI_THIRD)


class MonoclinicCell(UnitCell):
    def __init__(self, a: float, b: float, c: float, beta: float):
        self.a = a
        self.b = b
        self.c = c
        self.beta = beta

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.b, self.c, PI_HALF, self.beta, PI_HALF)


class TriclinicCell(UnitCell):
    def __init__(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def shape_tensor(self) -> npt.NDArray:
        return shape_tensor_3d(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)
