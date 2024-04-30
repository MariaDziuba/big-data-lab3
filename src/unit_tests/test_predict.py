import os
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from predict import Predictor
from conftest import db

def test_predictor():
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['vectorizer']['path_to_vectorizer_ckpt'])
    path_to_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['model']['path_to_model_ckpt'])

    predictor = Predictor()
    predictor.predict(db, "tmp_submission", "tmp_test", path_to_model_ckpt, path_to_vectorizer_ckpt)
    assert db.table_exists("tmp_submission")['result'].iloc[0]

    submission_tmp = db.read_table("tmp_submission")
    assert submission_tmp.iloc[0]["Category"] == "business" and submission_tmp.iloc[0]["ArticleId"] == 1, "Wrong prediction in line 1"
    assert submission_tmp.iloc[1]["Category"] == "sport" and submission_tmp.iloc[1]["ArticleId"] == 2, "Wrong prediction in line 2"
