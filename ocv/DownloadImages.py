import urllib.request
import cv2
import numpy as np
import os


def store_raw_images():
    neg_images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03901750'
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()
    pic_num = 1
    loc = '/home/kb/learn/chess/pawns/neg/'

    if not os.path.exists(loc):
        os.makedirs(loc)

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, loc + str(pic_num) + ".jpg")
            img = cv2.imread(loc + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            # should be larger than samples / pos pic (so we can place our image on it)
            resized_image = cv2.resize(img, (100, 100))
            cv2.imwrite(loc + str(pic_num) + ".jpg", resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

store_raw_images()