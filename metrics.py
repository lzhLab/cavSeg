import torch 
#import torchmetrics
#from torchmetrics.functional import dice_score, precision_recall
import torch.nn as nn
#import GeodisTK
#from scipy import ndimage
from sklearn import metrics
import numpy as np
from medpy import metric

def hausdorff95(res, ref):
    return metric.binary.hd95(res, ref)

def Assd(res, ref):
    return metric.binary.assd(res, ref)

def asd(res, ref):
    return metric.binary.asd(res, ref)

def hausdorff(res, ref):
    return metric.binary.hd(res, ref)

def voe(res, ref):
    return 1.-metric.binary.jc(res, ref)

def rvd(res, ref):
    return metric.binary.ravd(res, ref)

def dice(res, ref):
    return metric.binary.dc(res, ref)

def precision(res, ref):
    return metric.binary.precision(res, ref)

def msd(res, ref):
    return np.mean((res-ref)**2)

def recall(res, ref):
    return metric.binary.recall(res, ref)

def sen(res, ref):
    return metric.binary.sensitivity(res, ref)

def spe(res, ref):
    return metric.binary.specificity(res, ref)

