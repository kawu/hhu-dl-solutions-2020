import torch


def atab(M: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Apply matrix `M` to each word vector in matrix `w` and
    add bias vector `b` to the result of each transformation.

    Example:

    Input:
    >>> M = torch.randn(4, 3)
    >>> b = torch.randn(4)
    >>> w = torch.randn(5, 3)

    Test output:
    >>> for i in range(len(w)):
    ...     x = torch.mv(M, w[i]) + b
    ...     y = atab(M, b, w)[i]
    ...     # Allow for numerical errors
    ...     assert (abs(x - y) < 1e-5).all()
    """
    # Make sure the shapes match
    assert M.dim() == w.dim() == 2
    assert M.shape[1] == w.shape[1]

    # Solution based on broadcasting: the bias vector `b` gets added to each
    # row in the matrix resulting from the matrix/matrix product
    return torch.mm(w, M.t()) + b
