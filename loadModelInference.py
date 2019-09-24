from DGCNN_embedding import DGCNN
import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from DGCNN_embedding import DGCNN
from mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from util import cmd_args, load_data
from main import Classifier

model = Classifier(nn.Module)
model_to_load = torch.load('./Saved_Models/MyModel.pt')
model.load_state_dict(model_to_load)
model.eval()
