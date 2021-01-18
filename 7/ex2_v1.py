import torch
import torch.nn as nn
from torch import Tensor

from data import load_data, create_encoders, encode_with
from modules import *
from ex1 import accuracy
from utils import train

#############################################
# DATASET
#############################################

# Load the datasets
raw_train = load_data("train.csv")
raw_dev = load_data("dev.csv")

# Create the encoders, encode the datasets
char_enc, lang_enc = create_encoders(raw_train)
enc_train = encode_with(raw_train, char_enc, lang_enc)
enc_dev = encode_with(raw_dev, char_enc, lang_enc)

# Report the size of the datasets
print(f'# train = {len(enc_train)}')
print(f'# dev = {len(enc_dev)}')

#############################################
# TRAINING LOSS
#############################################

# You solved that one already, see for instance:
# https://github.com/kawu/hhu-dl-solutions-2020/blob/main/4/ex2.py
def loss(pred: Tensor, target: Tensor) -> Tensor:
    """Calculate the cross-entropy loss between the predicted and
    the target tensors, in case where the target is a scalar value
    (tensor of dimension 0).
    """
    # Preliminary checks (optional)
    assert pred.dim() == 1      # vector
    assert target.dim() == 0    # scalar
    assert 0 <= target.item() < pred.shape[0]
    nn_loss = nn.CrossEntropyLoss()
    return nn_loss(pred.view(1, -1), target.view(1))

#############################################
# MODEL
#############################################

model = nn.Sequential(
    nn.Embedding(char_enc.size()+1, 25, padding_idx=char_enc.size()),
    SimpleLSTM(25, 300),
    Pure(lambda xs: xs[-1]),
    nn.Linear(300, lang_enc.size()),
)

#############################################
# TRAINING
#############################################

train(model, loss, accuracy, enc_train, enc_dev, epoch_num=3,
        learning_rate=0.001, report_rate=1)
train(model, loss, accuracy, enc_train, enc_dev, epoch_num=2,
        learning_rate=0.0001, report_rate=1)
# => @1: loss(train)=14411.733, acc(train)=0.822, acc(dev)=0.808
# => @2: loss(train)=9076.391, acc(train)=0.871, acc(dev)=0.824
# => @3: loss(train)=6773.723, acc(train)=0.911, acc(dev)=0.829
# => @1: loss(train)=3628.002, acc(train)=0.952, acc(dev)=0.845
# => @2: loss(train)=2555.633, acc(train)=0.963, acc(dev)=0.845

#############################################
# EVALUATION
#############################################

# Put the model in the evaluation mode
model.eval()

def predict(model, name: str) -> str:
    """Language prediction with the trained model."""
    x = torch.tensor([char_enc.encode(char) for char in name])
    pred_y = torch.argmax(model(x), dim=0)
    return lang_enc.decode(pred_y.item())   # type: ignore

# First show the results for a selection of person names
print('# PREDICTION FOR SELECTED NAMES')
for name, gold_lang in raw_dev[:50]:
    pred_lang = predict(model, name)
    print(f'{name} =>\t{pred_lang}\t(gold: {gold_lang})')

# NOTE: Do not change the code below!
print('# FINAL EVALUATION')
dev_acc = accuracy(model, enc_dev)
if dev_acc > 0.8:
    print(f'PERFECT: acc(dev) = {dev_acc} (> 0.8)')
elif dev_acc > 0.7:
    print(f'GOOD: acc(dev) = {dev_acc} (> 0.7)')
elif dev_acc > 0.6:
    print(f'SUFFICIENT: acc(dev) = {dev_acc} (> 0.6)')
else:
    print(f'POOR: acc(dev) = {dev_acc} (<= 0.6)')
