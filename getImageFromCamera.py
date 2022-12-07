## Import Modules

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from keras.utils import load_img
from PIL import Image
from keras.models import load_model

md = load_model('models/age_model.h5')


def predict_age(url):
    img = load_img(url, grayscale=True)  # load ảnh dưới dạng grayscale thay vì rgb vì nếu sd rgb ta sẽ có 3 dimension (chiều) khiến ta phải load nhiều hơn trong khi bộ nhớ hạn ch
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255.0
    pred = md.predict(img.reshape(1, 128, 128, 1))
    pred_age = round(pred[0][0])
    return pred_age

#print(predict_age('user.jpg'))

import cv2
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)

while True:
    ret,frame=capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in faces:
        f = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # crop_image = frame[y: y + h + 5, x: x + w + 5]
        crop_image = frame[x: x + w, y: y + h]
        cv2.imwrite('user.jpg', crop_image)
    frame = cv2.putText(frame,'Age predict: ' +  str(predict_age('user.jpg')), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA )
    cv2.imshow('Age Prediction',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
