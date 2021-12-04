#capture image and save to data/image path
#if path has images in it delete and save new image in there
#Done - call detect on new image
#Done - call error detect on new image
import pandas as pd
import os
import cv2


#clears file
for filename in os.listdir("data/images"):
    os.remove(os.path.join("data/images", filename))


img_name = "data/images/image.jpg"

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

ret, frame = cam.read()
if not ret:
    print("failed to grab frame")
else:

    cv2.imwrite(img_name, frame)
    print("{} written!".format(img_name))

cam.release()

cv2.destroyAllWindows()



from detect import run
run()
#import ErrorDetection
from ErrorDetection import pred_images
# Storing the old scenarios in dq so everything prints out in the txt file one after another
dq=[]

#orderNumber1 = order(2, 4, 6)
#orderNumber2 = order(1, 0, 0)
#orderNumber3 = order(1, 0, 0)

#true_count = {"I":orderNumber3.I, "L":orderNumber3.L, "T":orderNumber3.T}

I = int(input("Enter the number of I shaped PVCs: "))
L = int(input("Enter the number of L shaped PVCs: "))
T = int(input("Enter the number of T shaped PVCs: "))

true_count = {"I":I,"L":L,"T":T}

dataf=dq

img_path = "data/images/opencv_frame_0.jpg"

pred_images(img_name,true_count,dataf)
# Put the data in the df and then output the df in a txt file. Basically we have everything stored in dataf
df = pd.DataFrame(dataf)
# Here dq has the old data and dataf has the new data
dq = dataf
df.to_csv("error.txt", index=False) 

