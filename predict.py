from keras.models import model_from_json
import cv2
import numpy as np
json_file=open("model.json",'r')
loaded_model=json_file.read()
json_file.close()
model=model_from_json(loaded_model)
model.load_weights("model.h5")
image=cv2.imread("Y19.JPG")#change the image to test the image you want
img=cv2.resize(image,(224,224))
imag=img.reshape(-1,224,224,3)
pred=model.predict(imag)
no=pred[0][0]
yes=pred[0][1]

print(no)
print(yes)
if(no>yes):
    cv2.putText(image,str(no),(50, 50),cv2.FONT_HERSHEY_SIMPLEX ,0.9,(0,255,0),2,cv2.LINE_AA)
    cv2.putText(image,"brain_tumor:negative",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
    print("negative")
if(yes>no):
    cv2.putText(image,str(yes),(50, 50),cv2.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(image,"brain_tumor:positive",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    print("positive")
print(pred)
cv2.imwrite('s.jpg',image)
cv2.imshow('s',image)
cv2.waitKey(0)
