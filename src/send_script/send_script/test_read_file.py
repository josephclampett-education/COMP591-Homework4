import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os

AREA_THRESHOLD = 100000

outPixelPositions = np.empty((4, 3))
outPhysicalPositions = np.array([[227.92, 693.22, 1.0], [677.43, 245.73, 1.0], [-102.37, 360.68, 1.0], [337.09, -91.91, 1.0]])

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
labelCount, labels = cv2.connectedComponents(inFrame_binary)

subtitle_texts = []
outFrame = np.copy(inFrame)
calIndex = 0
for label in range(1, labelCount): # Start from 1 to skip the background (label 0)
    mask = (labels == label).astype("uint8") * 255
    
    M = cv2.moments(mask)

    area = M["m00"]
    if area < AREA_THRESHOLD:
        continue

    Cx = M["m10"] / M["m00"]
    Cy = M["m01"] / M["m00"]

    principal = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])

    outPixelPositions[calIndex] = np.array([Cx, Cy, 1.0])
    calIndex = calIndex + 1

    # Draw Debug Info
    DrawDebugMarker(outFrame, area, label, Cx, Cy, principal)

calibrationMatrix, residuals, rank, singularValues = np.linalg.lstsq(outPixelPositions, outPhysicalPositions)
np.save('debug/CALIBRATION_MATRIX', calibrationMatrix)

# cv2.imshow("image", outFrame)
# cv2.waitKey(0)