#!/usr/bin/env python

import rclpy
import cv2
import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')
from tm_msgs.msg import *
from tm_msgs.srv import *

# arm client
def send_script(script):
    arm_node = rclpy.create_node('arm')
    arm_cli = arm_node.create_client(SendScript, 'send_script')

    while not arm_cli.wait_for_service(timeout_sec=1.0):
        arm_node.get_logger().info('service not availabe, waiting again...')

    move_cmd = SendScript.Request()
    move_cmd.script = script
    arm_cli.call_async(move_cmd)
    arm_node.destroy_node()

# gripper client
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

    #--- move command by joint angle ---#
    # Break robot any% speedrun
    # script = 'PTP(\"JPP\",45,0,90,0,90,0,35,200,0,false)'

    #--- move command by end effector's pose (x,y,z,a,b,c) ---#
    # targetP1 = "398.97, -122.27, 748.26, -179.62, 0.25, 90.12"s

    # Initial camera position for taking image (Please do not change the values)
    # For right arm: targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    # For left  arm: targetP1 = "350.00, 350, 730, -180.00, 0.0, 135.00"
    targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    send_script(script1)

    # targetP1 = "230.00, 230, 110, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "230.00, 230, 730, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "100.00, 230, 730, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "100.00, 230, 110, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "100.00, 230, 730, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "230.00, 100, 730, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "230.00, 100, 110, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # targetP1 = "230.00, 100, 730, -180.00, 0.0, 135.00"
    # script1 = "PTP(\"CPP\"," + targetP1+",100,200,0,false)"
    # send_script(script1)

    # # Z = 100 should be the table
    # # c is what we need to rotate
    # targetP1 = "230.00, 230, 101, -180.00, 0.0, 100.00"
    # script1 = f"PTP(\"CPP\",{targetP1},100,200,0,false)"
    # send_script(script1)

    # FOUR CORNERS (CC)
    # Orange TR (680.93, 243) 
    # Orange BR (333, -83.4)
    # Yellow BL (-103.44, 357.61)
    # Red TL (230.41, 704.27)

    # DIMENSION OF BLOCK
    # 1"

# What does Vision_DoJob do? Try to use it...
# -------------------------------------------------
    send_script("Vision_DoJob(job1)")
    cv2.waitKey(1)
#--------------------------------------------------
    
    # set_io(1.0) # 1.0: close gripper, 0.0: open gripper
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    


    
