#!/usr/bin/env python
import rclpy

from rclpy.node import Node

import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *
from cv_bridge import CvBridge
import cv2
import math
import numpy as np
import os
from sensor_msgs.msg import Image

AREA_THRESHOLD = 100000

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

def SendSetOrientationScript(x, y, z, theta):
    positionInfo = f"{x}, {y}, {z}, -180.00, 0.0, {135.0 + theta}"
    outScript = f"PTP(\"CPP\",{positionInfo},100,200,0,false)"
    send_script(outScript)

class ImageSub(Node):
    def __init__(self, nodeName):
        super().__init__(nodeName)
        self.subscription = self.create_subscription(Image, 'techman_image', self.image_callback, 10)
    
    def image_callback(self, data):
        self.get_logger().info('Received image')

        bridge = CvBridge()
        inFrame = bridge.imgmsg_to_cv2(data)
        
        CalibrationMatrix = np.load("src/debug/CALIBRATION_MATRIX.npy")

        cv2.imwrite(f"src/debug/image.jpg", inFrame)

        inFrame_grayscale = cv2.cvtColor(inFrame, cv2.COLOR_BGR2GRAY)
        inFrame_blurred = cv2.GaussianBlur(inFrame_grayscale, (5, 5), 0)
        _, inFrame_binary = cv2.threshold(inFrame_blurred, 140, 255, cv2.THRESH_BINARY)
        labelCount, labels = cv2.connectedComponents(inFrame_binary)

        subtitle_texts = []
        outFrame = np.copy(inFrame)
        for label in range(1, labelCount): # Start from 1 to skip the background (label 0)
            mask = (labels == label).astype("uint8") * 255
            
            M = cv2.moments(mask)

            area = M["m00"]
            if area < AREA_THRESHOLD:
                continue

            Cx = M["m10"] / M["m00"]
            Cy = M["m01"] / M["m00"]

            principal = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])

            # Draw Debug Info
            DrawDebugMarker(outFrame, area, label, Cx, Cy, principal)

            # Target
            physicalPosition = np.dot(np.array([Cx, Cy, 1.0]), CalibrationMatrix)
            xPP = np.clip(physicalPosition[0], -103, 680)
            yPP = np.clip(physicalPosition[1], -83, 704)

            set_io(0.0)
            SendSetOrientationScript(xPP, yPP, 730)
            set_io(1.0)
            SendSetOrientationScript(xPP, yPP, 710)
            set_io(0.0)
        
        cv2.imwrite(f"src/debug/image_annotated.jpg", outFrame)

def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

def set_io(state):
    gripper_node = rclpy.create_node('gripper')
    gripper_cli = gripper_node.create_client(SetIO, 'set_io')

    while not gripper_cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not availabe, waiting again...')
    
    io_cmd = SetIO.Request()
    io_cmd.module = 1
    io_cmd.type = 1
    io_cmd.pin = 0
    io_cmd.state = state
    gripper_cli.call_async(io_cmd)
    gripper_node.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSub('image_sub')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
