# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:54:34 2020

@author: oguzkaya
"""


    
from keras.models import model_from_json
import numpy as np

class FacialExpressionModel(object):

    EMOTIONS_LIST = ["G", "K",
                     "O", "W"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
