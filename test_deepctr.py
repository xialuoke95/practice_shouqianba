#coding: utf-8

import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import deepctr
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat,get_feature_names

from pympler import muppy, summary, asizeof, tracker

# https://stackoverflow.com/questions/45882401/how-to-deal-with-userwarning-converting-sparse-indexedslices-to-a-dense-tensor
# maybe should use? tf.gather