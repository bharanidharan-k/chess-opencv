import cv2
import os

def convertoToGray():
    count = 0
    loc = '/home/kb/learn/chess/pawns/pos/original'
    destLoc = '/home/kb/learn/chess/pawns/pos'
    for file_type in [loc]:
        for img in os.listdir(file_type):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    original = cv2.imread(current_image_path)
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                    original = cv2.resize(original, (50,50))
                    cv2.imwrite(destLoc+'/cb'+ str(count)+'.jpg', original)
                    count+=1
                except Exception as e:
                    print(str(e))

convertoToGray()