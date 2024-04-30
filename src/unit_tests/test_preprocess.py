import os
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from preprocess import Preprocessor
import pandas as pd
from conftest import db

def test_preprocessor():
    preprocessor = Preprocessor()
    is_test_false = preprocessor.load_and_preprocess_data(db, "tmp_test", isTest=False)
    is_test_true = preprocessor.load_and_preprocess_data(db, "tmp_test", isTest=True)    

    assert type(is_test_false) is tuple
    assert type(is_test_true) is list
    X_train, y_train = is_test_false
    assert X_train[0] == 'business good' and X_train[1] == 'football popular sport'
