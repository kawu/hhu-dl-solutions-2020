import torch
import torch.nn as nn


class Replace(nn.Module):

    """A replacement module which, given a (vector) tensor of integer indices,
    replaces each value in the vector with a certain pre-specified index value `ix`,
    with a certain pre-specified probability `p`.

    Examples:
    
    Create the module with replacement probability 0.5 and the special
    index equal to 5
    >>> frg = Replace(p=0.5, ix=5)

    Check if the module preserves the shape of the input matrix
    >>> x = torch.tensor([0, 1, 0, 3, 4, 2, 3])
    >>> frg(x).shape == x.shape
    True

    When `p` is set to 0, the module should behave as an identity function
    >>> frg = Replace(p=0.0, ix=5)
    >>> (frg(x) == x).all().item()
    True

    When `p` is set to 1, all values in the input tensor should
    be replaced by `ix`
    >>> frg = Replace(p=1.0, ix=5)
    >>> (frg(x) == 5).all().item()
    True

    In the evaluation mode, the module should also behave as an identity,
    whatever the probability `p`
    >>> frg = Replace(p=0.5, ix=5)
    >>> _ = frg.eval()
    >>> (frg(x) == x).all().item()
    True

    Make sure the module is actually non-deterministic and returns
    different results for different applications
    >>> frg = Replace(p=0.5, ix=5)
    >>> x = torch.randint(5, (20,))    # length 20, values in [0, 5)
    >>> results = set()
    >>> for _ in range(1000):
    ...     results.add(frg(x))
    >>> assert len(results) > 100

    See if the number of special index values the resulting tensor
    contains on average is actually close to 0.5 * len(x)
    >>> special_ix_num = [(y == 5).sum().item() for y in results]
    >>> avg_special_ix_num = sum(special_ix_num) / len(special_ix_num)
    >>> 0.5*len(x) - 0.5 <= avg_special_ix_num <= 0.5*len(x) + 0.5
    True
    """

    def __init__(self, p: float, ix: int):
        super().__init__()
        self.repl_ix = ix
        self.p = p

    def forward(self, ixs):
        if self.training:
            assert ixs.dim() == 1
            mask = (torch.empty_like(ixs, dtype=torch.float).uniform_() > self.p).long()
            unmask = 1 - mask    # XOR
            return ixs*mask + self.repl_ix*unmask
        else:
            return ixs
