import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os,ssl,time
from PIL import Image
import cv2
import PIL.ImageOps
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if not os.environ.get("PYTHONHTTPSVERIFY","")and getattr(ssl, "_create_unverified_context",None):
    ssl._create_default_https_context=ssl._create_unverified_context

x,y=fetch_openml("mnist_784",version=1,return_X_y=True)
print(pd.Series(y).value_counts())
classes=["0","1","2","3","4","5","6","7","8","9"]
mclasses=len(classes)

x_test,x_train,y_test,y_train=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scale=x_train/255.0
x_test_scale=x_test/255.0

clf=LogisticRegression(solver="saga", multi_class="multinomial").fit(x_train_scale,y_train)
ypredict=clf.predict(x_test_scale)
accuracy=accuracy_score(y_test,ypredict)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        #Whatever capture is taken will be read and returned
        ret,frame = cap.read()
        #make grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #find the rows and columns of the image
        height, width = gray.shape
        #find the start and end of the columns and rows
        upper_left = (int(width / 2 -56),int(height / 2 -56))
        bottom_right= (int(width / 2 +56),int(height / 2 +56))
        #drawing the rectangle
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)
        #access metrics has 4 parameters[start row:end row,start column:end column]
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
        #selecting only roi (region of interest) to be used
        im_pil = Image.fromarray(roi)
        #make the image flipped
        image_bw = im_pil.convert("L")
        #resizing the image to 28x28 image
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
        #reversing the image with filter
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        #size of pixel filter
        pixel_filter = 20
        #finding the percentile - how much to flip
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        #helps us to avoid issues like prediction going beyond meaningful bounds (to limit values of an array to a specified range)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, min_pixel,0,255)
        #taking image with maximum value
        max_pixel = np.max(image_bw_resized_inverted)
        #change the data into an array to use in the model for prediction
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        #converting a 1d array
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        #predicting the output
        test_pred = clf.predict(test_sample)
        #printing result
        print("Predicted class is:", test_pred)
        #display the image
        cv2.imshow("frame",gray)
        #to stop the output
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        #if any errors we use exception
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
