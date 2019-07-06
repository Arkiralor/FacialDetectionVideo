# Classifier set provided by OpenCV: https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import time as tm

first_frame = None
status_list = [None, None]
times = []
df=pd.DataFrame(columns=["Start", "End"])
ctm = tm.time()

key = -1

vid = cv2.VideoCapture(0)
while True:
    check, frame = vid.read()
    status = 0
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)

    if first_frame is None:
        first_frame = grey
        continue

    delta_frame = cv2.absdiff(first_frame, grey)
    thresh_delta = cv2.threshold(delta_frame, 43, 255, cv2.THRESH_BINARY) [1]
    thresh_delta = cv2.erode(thresh_delta, None, iterations = 1)
    thresh_delta = cv2.dilate(thresh_delta, None, iterations = 1)
    cnts,_ = cv2.findContours(thresh_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in cnts:
        if cv2.contourArea(contour) < 1400:
            continue

        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    status_list=status_list[-2:]

    if status_list[-1]==1 and status_list[-2]==0:
        times.append(float(tm.time()-ctm))
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(float(tm.time()-ctm))


    cv2.imshow('Compared Image', grey)
    cv2.imshow('Comparing Image', delta_frame)
    cv2.imshow('Differential Image', thresh_delta)
    cv2.imshow('Detected Motion', frame)


    key = cv2.waitKey(17)
    if key == ord('q'):
        vid.release()
        cv2.destroyWindow('Detected Motion')
        cv2.destroyWindow('Compared Image')
        cv2.destroyWindow('Comparing Image')
        cv2.destroyWindow('Differential Image')
        break
print(status_list)
for i in range(0, len(times), 2):
    if i in range(len(times)-1):
        print(times[i], times[i+1])

for i in range(0, len(times)-1, 2):
    if i in range(len(times) - 1):
        df=df.append({"Start": times[i], "End": times[i+1]}, ignore_index = True)

    df.to_csv("Output/Time Map.csv")



data = pd.read_csv("Output/Time Map.csv")
AxX = data["Start"].values
AxY = data["End"].values - data["Start"].values

print(AxX)
print(AxY)

plt.bar(AxX, AxY, width=AxY, color='blue', align='edge')
plt.xlabel('Detected At --->')
plt.ylabel('Duration --->')

plt.savefig("Output/Motion Detection Graph_0.png", dpi=160, facecolor='w', edgecolor='w', orientation='landscape', papertype='a4', format='png', transparent=False)

plt.show()
if key == ord('q'):
    cv2.destroyAllWindows()

