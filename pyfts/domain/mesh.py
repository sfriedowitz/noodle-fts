import math


class Mesh:
    """Discrete grid for evaluating the system."""

    def __init__(self, *dimensions: int):
        super().__init__()

        if not 1 <= len(dimensions) <= 3:
            raise ValueError("Mesh can only have 1, 2, or 3 dimensions.")

        self.dimensions: list[int] = list(dimensions)

    def __repr__(self):
        dims_str = ", ".join(str(x) for x in self.dimensions)
        return f"Mesh({dims_str})"

    @property
    def ndim(self) -> int:
        return len(self.dimensions)

    @property
    def size(self) -> int:
        return math.prod(self.dimensions)

    def kmesh(self) -> "Mesh":
        match len(self.dimensions):
            case 1:
                return Mesh(self.dimensions[0] / 2 + 1)
            case 2:
                return Mesh(self.dimensions[0], self.dimensions[1] / 2 + 1)
            case 3:
                return Mesh(*self.dimensions[:2], self.dimensions[2] / 2 + 1)
