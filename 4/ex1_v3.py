import torch
import torch.nn as nn


# For reproduction
torch.manual_seed(seed=0)


def create_model(alpha_size: int, emb_size: int, class_num: int) -> nn.Module:
    """Construct a neural language prediction model.

    Arguments:
    * alpha_size: alphtabet size (number of different characters)
    * emb_size: size of vector representations assigned to the different characters
    * class_num: number of output classes (languages)

    Output: A neural model which transforms a tensor vector with
    input character indices, such as `tensor([0, 1, 2, 1])`, to
    a score vector, such as `tensor([0.6861, -0.7673,  1.2969])`. 
    
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

    class Squeeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, v):
            return v.squeeze(0)
    

    class Unsqueeze(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, v):
            return v.unsqueeze(0)


    model = nn.Sequential(
        Unsqueeze(),
        nn.EmbeddingBag(alpha_size, emb_size, mode="sum"),
        nn.Linear(emb_size, class_num),
        Squeeze()
    )

    return model