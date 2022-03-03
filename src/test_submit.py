import pandas as pd
import numpy as np


def eval(f):
    test = pd.read_csv('../inputs/dataset/example_test.csv')
    ss = pd.read_csv('../inputs/dataset/example_sample_submission.csv')
    ss.target = f(test)

    return ss


