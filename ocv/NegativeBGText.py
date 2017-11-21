import cv2
import os

def create_pos_n_neg():
    for file_type in ['/home/kb/learn/chess/pawns/neg']:
        for img in os.listdir(file_type):

            # positive is not needed, because, we ca use create samples
            # if file_type == 'pos':
            #     line = file_type + '/' + img + ' 1 0 0 50 50\n'
            #     with open('info.dat', 'a') as f:
            #         f.write(line)
            if file_type == '/home/kb/learn/chess/pawns/neg':
                line = file_type + '/' + img + '\n'
                with open('/home/kb/learn/chess/pawns/neg/bg.txt', 'a') as f:
                    f.write(line)

create_pos_n_neg()