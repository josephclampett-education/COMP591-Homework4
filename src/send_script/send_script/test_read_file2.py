import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

AREA_THRESHOLD = 100000

outPixelPositions = np.empty((4, 3))
outPhysicalPositions = np.array([[230.41, 704.27, 1.0], [680.93, 243, 1.0], [-103.44, 357.61, 1.0], [333, -83.4, 1.0]])

subtitle_texts = []
def DrawDebugMarker(frame, area, label, Cx, Cy, theta):
    ### Draw the principal axis ###
    cv2.circle(frame, (int(Cx), int(Cy)), 5, (0, 255, 0), -1)

    ### Draw the principal axis ###
    line_length = int(np.sqrt(area) * 0.1)
    # Calculate the startpoint of the principal axis
    x1 = int(Cx - line_length * math.cos(theta))
    y1 = int(Cy - line_length * math.sin(theta))
    # Calculate the endpoint of the principal axis
    x2 = int(Cx + line_length * math.cos(theta))
    y2 = int(Cy + line_length * math.sin(theta))

    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the object number next to the centroid
    cv2.putText(frame, str(label), (int(Cx) - 20, int(Cy) + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # Collect the information for the subtitle
    subtitle_texts.append(f"Object {label}: Centroid: ({Cx:.2f}, {Cy:.2f}), Angle: {(theta * (180 / math.pi)):.2f} deg")

inFrame = cv2.imread(f"debug/image.jpg")

inFrame_grayscale = cv2.cvtColor(inFrame, cv2.COLOR_BGR2GRAY)
inFrame_blurred = cv2.GaussianBlur(inFrame_grayscale, (5, 5), 0)
_, inFrame_binary = cv2.threshold(inFrame_blurred, 140, 255, cv2.THRESH_BINARY)

contours = cv2.findContours(inFrame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

subtitle_texts = []
outFrame = np.copy(inFrame)
calIndex = 0
for i in range(1, len(contours)): # Start from 1 to skip the background (label 0)    
    
    rotrect = cv2.minAreaRect(contours[i])
    
    area = rotrect[1][0] * rotrect[1][1]
    if area < 2000:
        continue

    angle = rotrect[-1]
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle
    angle = np.deg2rad(angle)

    box = cv2.boxPoints(rotrect)
    box = np.intp(box)
    cv2.drawContours(outFrame, [box], 0, (0, 0, 255), 2)

    outPixelPositions[calIndex] = np.array([rotrect[0][0], rotrect[0][1], 1.0])
    calIndex = calIndex + 1

    # Draw Debug Info
    DrawDebugMarker(outFrame, 1000000, calIndex, rotrect[0][0], rotrect[0][1], angle)

calibrationMatrix, residuals, rank, singularValues = np.linalg.lstsq(outPixelPositions, outPhysicalPositions)
# np.save(f"debug/CALIBRATION_MATRIX", calibrationMatrix)

cv2.imshow("image", outFrame)
cv2.waitKey(0)