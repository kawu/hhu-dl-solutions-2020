import torch
import torch.nn as nn


class DiscardIrrelevant(nn.Module):

    def __init__(self, alpha_size: int):
        super().__init__()
        self.n = alpha_size
    
    def forward(self, ixs: torch.Tensor) -> torch.Tensor:
        return ixs[(ixs >= 0) & (ixs < self.n)]


class CBOW(nn.Module):

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return torch.sum(m, dim=0)


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
    
    # The model should ignore negative character indices (e.g. `-1`)
    # as well as indices >= `alpha_size`.
    >>> x1 = torch.tensor([0, 5])
    >>> x2 = torch.tensor([-5, 0, 5, 10])
    >>> (model(x1) == model(x2)).all().item()
    True
    """
    return nn.Sequential(
        DiscardIrrelevant(alpha_size),
        nn.Embedding(alpha_size, emb_size),
        CBOW(),
        nn.Linear(emb_size, class_num)
    )
