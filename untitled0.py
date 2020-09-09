# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 23:05:28 2020

@author: oguzkaya
"""


from imageai.Detection import VideoObjectDetection
import os


base_path_train = 'dataset/training_set'
base_path_test = 'dataset/test_set'
base_path_video = 'dataset'
execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(base_path_video , "yolo.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join( base_path_video, "test_video.mp4"),
                                output_file_path=os.path.join(base_path_video, "test_result_1")
                                , frames_per_second=29, log_progress=True)
print(video_path)