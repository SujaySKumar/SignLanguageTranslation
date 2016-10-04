import cv2
import os, pickle
import numpy as np
from matplotlib import pyplot as plt
from feature_vector_gen import get_feature_vector

class GenerateTraining:

    def __init__(self, path):
        self.path_to_dataset = path

    def generate_training_data(self):
        words = os.listdir(self.path_to_dataset)
        with open('word_to_vec.pkl','rb') as fp:
            word_to_vec = pickle.load(fp)

        final_training_data = []
        count = 0
        for word in words:
            print "Generating traning data for the word ",word
            print "Count of the word is ", count
            count+=1
            target_vec = word_to_vec[word]
            temp_path = self.path_to_dataset+'/'+word
            videos = os.listdir(temp_path)
            for video in videos:
                input_vec = get_feature_vector(temp_path+'/'+video)
                final_training_data.append([input_vec, target_vec])
        return final_training_data

g = GenerateTraining('/home/sujay/Desktop/SLT/dataset100')
t_data = g.generate_training_data()
print "Done with training"
with open('training_data.pkl','wb') as fp:
    pickle.dump(t_data, fp)
import ipdb;ipdb.set_trace()
