import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):

    '''
    Contextualise the input sequence of embedding vectors using unidirectional
    LSTM.

    Type: Tensor[N x Din] -> Tensor[N x Dout], where
    * `N` is is the length of the input sequence
    * `Din` is the input embedding size
    * `Dout` is the output embedding size

    Example:

    >>> lstm = SimpleLSTM(3, 5)   # input size 3, output size 5
    >>> xs = torch.randn(10, 3)   # input sequence of length 10
    >>> ys = lstm(xs)             # equivalent to: lstm.forward(xs)
    >>> list(ys.shape)
    [10, 5]
    '''

    def __init__(self, inp_size: int, out_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=inp_size, hidden_size=out_size)

    def forward(self, xs):
        '''Apply the LSTM to the input sequence.

        Arguments:
        * xs: a tensor of shape N x Din, where N is the input sequence length
            and Din is the embedding size

        Output: a tensor of shape N x Dout, where Dout is the output size
        '''
        # Apply the LSTM, extract the first value of the result tuple only
        out, _ = self.lstm(xs.view(xs.shape[0], 1, xs.shape[1]))
        # Reshape and return
        return out.view(out.shape[0], out.shape[2])


class SameConv(nn.Module):

    """The class implements the so-called ,,same'' variant of a 1-dimensional
    convolution, in which the length of the output sequence is the same as the
    length of the input sequence.

    The embedding size, often called the number of ,,channels'' in the context
    of convolution, can change (as specified by the hyper-parameters).
    """

    def __init__(self, inp_size: int, out_size: int, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)

    def forward(self, x):
        # As usual, we have to account for the batch dimension.  On top of
        # that, the convolution requires that the sentence dimension and the
        # embedding dimension are swapped.
        x = x.t().view(1, x.shape[1], x.shape[0])
        # Pad the input tensor on the left and right with 0's.  If the kernel
        # size is odd, the padding on the left is larger by 1.
        padding = (
            self.kernel_size // 2,
            (self.kernel_size - 1) // 2,
        )
        out = self.conv(F.pad(x, padding))
        out_reshaped = out.view(out.shape[1], out.shape[2]).t()
        return out_reshaped


#############################################
# Other modules, for the record (not used
# in the final solution)
#############################################


class Pure(nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Max(nn.Module):

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.max(m, dim=0).values


class SimpleConv(nn.Module):

    """The simple variant of the convolution.

    The length of the output sequence will be smaller than the length of the
    input sequence (if kernel size > 1).
    """

    def __init__(self, inp_size: int, out_size: int, kernel_size=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(inp_size, out_size, kernel_size)

    def forward(self, x):
        # As usual, we have to account for the batch dimension.  On top
        # of that, the convolution requires that the sentence dimension and
        # the embedding dimension are swapped.
        x = x.view(1, x.shape[1], x.shape[0])
        out = self.conv(x)
        out_reshaped = out.view(out.shape[2], out.shape[1])
        return out_reshaped


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

