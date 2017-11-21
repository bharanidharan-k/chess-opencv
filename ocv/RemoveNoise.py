import cv2
import os
import numpy as np

def remove_noise():
    match = False
    loc = '/home/kb/learn/chess/pawns/neg'
    noiseLoc = '/home/kb/learn/chess/pawns/noise'
    for file_type in [loc]:
        for img in os.listdir(file_type):
            for noise in os.listdir(noiseLoc):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread(noiseLoc+'/'+str(noise))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))

remove_noise()