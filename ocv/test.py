import cv2

cv2.namedWindow("preview")
# /home/kb/anaconda3/share/OpenCV/haarcascades
frontalFaceCascade = cv2.CascadeClassifier('/home/kb/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
vc = cv2.VideoCapture(0)
vc.set(3,640)
vc.set(4,480)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = frontalFaceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=3)
    print(len(faces))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()