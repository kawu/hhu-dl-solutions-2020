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

    def __init__(self, p: float = 0.5, ix: int = 5) -> None:
        super(Replace, self).__init__()
        self.p = p
        self.ix = ix

    def forward(self, x, inplace: bool = False) -> torch.Tensor:

        # return unmodified x on eval
        if not self.training:
            return x

        if inplace is False:
            x = x.clone()

        # create normal distribution tensor with size x, filled with 0,1
        mask: torch.Tensor = torch.distributions.Bernoulli(
            probs=(1 - self.p)
        ).sample(x.size())

        # apply bool mask replacing to x
        x[~mask.bool()] = self.ix

        return x
