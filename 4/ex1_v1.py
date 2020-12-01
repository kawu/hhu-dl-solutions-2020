import torch
import torch.nn as nn


# For reproduction (optional)
torch.manual_seed(seed=0)


class CBOW(nn.Module):

    # NOTE: In this special case we can actually skip __init__
    # def __init__(self):
    #     super().__init__()
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # NOTE: Alternatively, you could sum the rows of `m` in a loop, but
        # that would be less efficient, especielly in the backward pass
        # (calculation of the gradient)
        return torch.sum(m, dim=0)
        # y = 0
        # for x in m:
        #     y += x
        # return y    # type: ignore


def create_model(alpha_size: int, emb_size: int, class_num: int):
    """Construct a neural language prediction model.

    Arguments:
    * alpha_size: alphtabet size (number of different characters)
    * emb_size: size of vector representations assigned to the different characters
    * class_num: number of output classes (languages)

    Output: A neural model which transforms a tensor vector with
    input character indices, such as `tensor([0, 1, 2, 1])`, to
    a score tensor, such as `tensor([0.6861, -0.7673,  1.2969])`. 
    
    Examples:

    # Create a language prediction model for 10 distinct characters
    # and 5 distinct languages
    >>> alpha_size = 10
    >>> emb_size = 5
    >>> class_num = 5
    >>> model = create_model(alpha_size, emb_size, class_num)

    # The model should be a nn.Module
    >>> isinstance(model, nn.Module)
    True

    # The size of the output score vector should be equal to `class_num`.
    >>> len(model(torch.tensor([0, 1, 2]))) == class_num
    True
    """
    return nn.Sequential(
        nn.Embedding(alpha_size, emb_size),
        CBOW(),
        nn.Linear(emb_size, class_num)
    )
