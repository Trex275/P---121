import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

cap = cv2.VideoCapture(0)

time.sleep(2)
img = 0

for i in range(60):
    ret, img = cap.read()

img = np.flip(img, axis = 1)

while(cap.isOpened()):
    ret, bg = cap.read()
    if not ret:
        break

    bg = np.flip(bg, axis = 1)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([30, 30, 0])
    upper_range = np.array([103, 153, 70])
    mask_1 = cv2.inRange(hsv, lower_range, upper_range)

    lower_range = np.array([30, 30, 0])
    upper_range = np.array([103, 153, 70])
    mask_2 = cv2.inRange(hsv, lower_range, upper_range)

    mask_1 = mask_1 + mask_2

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    res_1 = cv2.bitwise_and(img, img, mask = mask_2)
    res_2 = cv2.bitwise_and(bg, bg, mask = mask_1)

    final_result = cv2.addWeighted(res_1, 1, res_2, 1, 0)
    output_file.write(final_result)
    cv2.imshow("magic", final_result)
    

cap.release()
cv2.destroyAllWindows()