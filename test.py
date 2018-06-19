import tensorflow as tf
import numpy as np
import scipy.io as sio
from six.moves import xrange
import time
import h5py
import pickle

import argparse
from utils import *
import model

parser = argparse.ArgumentParser()

parser.add_argument('--hiddendim', default = 50, type = int)
parser.add_argument('-f', '--featurefile', default = 'scapefeature.mat', type = str)
parser.add_argument('-n', '--neighbourfile', default = 'scapeneighbour.mat', type = str)
parser.add_argument('--neighbourvariable', default = 'neighbour', type = str)
# parser.add_argument('-d', '--distancefile', default = 'scapedistance.mat', type = str)
# parser.add_argument('--distancevariable', default = 'distance', type = str)
parser.add_argument('--l1', default = 0.5, type = float)
parser.add_argument('--l2', default = 0.5, type = float)
parser.add_argument('--lr', default = 0.001, type = float)
parser.add_argument('--finaldim', default = 9, type = int)
parser.add_argument('-l', '--layers', default = 1, type = int)
# parser.add_argument('-m', '--maxepoch', default=2000, type = str)
parser.add_argument('--modelfile', default = 'convmesh-model-2000', type = str)

args = parser.parse_args()

hidden_dim = args.hiddendim
featurefile = args.featurefile
neighbourfile = args.neighbourfile
neighbourvariable = args.neighbourvariable
# distancefile = args.distancefile
# distancevariable = args.distancevariable
lambda1 = args.l1
lambda2 = args.l2
lr = args.lr
finaldim = args.finaldim
layers = args.layers
modelfile = args.modelfile
# maxepoch = args.maxepoch

feature, logrmin, logrmax, smin, smax, pointnum = load_data(featurefile)

neighbour, degrees, maxdegree = load_neighbour(neighbourfile, neighbourvariable, pointnum)

# geodesic_weight = load_geodesic_weight(distancefile, distancevariable, pointnum)

model = model.convMESH(pointnum, neighbour, degrees, maxdegree, hidden_dim, finaldim, layers, lambda1, lambda2, lr)

model.individual_dimension(modelfile, feature, logrmin, logrmax, smin, smax)