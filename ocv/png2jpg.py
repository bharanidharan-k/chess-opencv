from glob import glob
import cv2
pngs = glob('/home/kb/learn/chess/pawns/pos/original/*.png')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)