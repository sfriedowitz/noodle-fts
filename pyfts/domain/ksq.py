import numpy as np

from pyfts.domain.cell import UnitCell
from pyfts.domain.mesh import Mesh


def ksq_grid(mesh: Mesh, cell: UnitCell) -> np.array:
    """Compute the squared spatial frequencies for the mesh and cell metric tensor.

    The output tensor is half size along its last dimension and has shape of `mesh.kshape`.
    """
    pass
    # metric_tensor = shape_tensor.T @ shape_tensor
    # metric_inv = metric_tensor.inverse()

    # kvec_list = []
    # for i, n in enumerate(self.shape):
    #     if i == self.ndim - 1:
    #         k = torch.fft.rfftfreq(n).to(metric_inv)
    #     else:
    #         k = torch.fft.fftfreq(n).to(metric_inv)
    #     kvec_list.append(2 * torch.pi * n * k)

    # kvecs = torch.cartesian_prod(*kvec_list)
    # kvecs = torch.atleast_2d(kvecs).view(-1, self.ndim)

    # ksq = ((kvecs @ metric_inv) * kvecs).sum(axis=-1)
    # return ksq.reshape(self.kshape)
