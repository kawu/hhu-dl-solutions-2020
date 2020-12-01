from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import data
from data import enc_data

from ex1_v1 import create_model
from ex2 import calculate_loss

# For reproducibility (optional)
torch.manual_seed(0)

# Create the language prediction model
# NOTE: Below, do not hard-code the alphabet size nor the number of classes;
# If the dataset changes, your code might break!
model = create_model(
    # TODO: provide appropriate alphabet size
    alpha_size=data.char_enc.size(),
    # TODO: choose appropriate embedding size
    emb_size=10,
    # TODO: provide the number of output classes
    class_num=data.lang_enc.size()
)

# TODO: Create optimizer (e.g. torch.optim.Adam) with appropriate arguments
optim = torch.optim.Adam(model.parameters())

# Perform SGD for a selected number of epochs
epoch_num = 50    # TODO: change to None
for k in range(epoch_num):
    for x, y in enc_data:
        # TODO: Calculate the loss, call backward
        # TODO: Apply the optimisation step
        calculate_loss(model(x), y).backward()
        optim.step()	# version of `nudge` provided by `Adam`

#################################################
# EVALUATION SECTION: DO NOT MODIFY
#################################################

# Let's verify the final losses
total_loss = sum(
    calculate_loss(model(x), y).item()
    for x, y in enc_data
)

# Evaluation: total loss should be smaller than 1
if total_loss < 1.0:
    print(f"OK: final total loss {total_loss} < 1")
else:
    print(f"FAILED: final total loss {total_loss} >= 1")