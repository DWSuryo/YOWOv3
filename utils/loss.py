import torch
from torch.nn.functional import cross_entropy, one_hot
import math
from utils.box import make_anchors
import cv2
import torch.nn.functional as F

# loss functions
from utils.lossfunction.tal import TAL
from utils.lossfunction.simota import SimOTA
from utils.lossfunction.normal import Normal

def build_loss(model, config):
    loss_type = config['loss']

    if loss_type == 'tal':
        return TAL(model, config)
    elif loss_type == 'simota':
        return SimOTA(model, config)
    elif loss_type == 'normal':
        return Normal(model, config)