# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:09:20 2020

@author: oguzkaya
"""

from imageai.Detection import ObjectDetection
import os
from time import time
models_path = "".join((os.getcwd().rstrip("examples"), "models"))
image_path = "".join((os.getcwd().rstrip("examples"), "images"))
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(models_path , "yolo.h5"))
detector.loadModel()
our_time = time()
detections = detector.detectObjectsFromImage(input_image=os.path.join(image_path , "image3.jpg"), output_image_path=os.path.join(image_path , "image3new.jpg"), minimum_percentage_probability=30)
print("IT TOOK : ", time() - our_time)
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"]  )
    print("--------------------------------")