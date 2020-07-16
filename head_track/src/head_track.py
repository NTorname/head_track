#!/usr/bin/env python


# TEST VERSION NOT FINISHED CODE


import rospy, cv2.aruco, numpy, cv2, sys, time, math, tf, sensor_msgs.msg, geometry_msgs.msg
from cv_bridge import CvBridge
from std_msgs.msg import Header as HeaderMsg
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import copy
import array

# define tag
id_to_find = 1
marker_size = 0.04  # m
id_back = 380
id_back_right = 403
id_right = 643
id_front_right_1 = 303
id_front_right_2 = 473
id_front = 688

# initialize these values here (will be in init)
tvec = [0, 0, 0]
q = [0, 0, 0, 1]
last_q = [0,0,0,1]
last_tvec = [0,0,0]

# get camera calibration
calib_path = '/home/csrobot/catkin_ws/src/head_track/calibration/'
camera_matrix = numpy.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
camera_distortion = numpy.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

# 180 degree matrix rotation around x axis   may not be necessary for this application
R_flip = numpy.zeros((3, 3), dtype=numpy.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# define aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create()

# set up link between ros topics and opencv
bridge = CvBridge()

# TODO set up image publisher
pubIm = rospy.Publisher("/detected_frame", sensor_msgs.msg.Image, queue_size=10)
pubPose = rospy.Publisher("/head_pose", geometry_msgs.msg.PoseStamped, queue_size=10)

pose = geometry_msgs.msg.PoseStamped()

def aa2quat(aa):
    angle = np.linalg.norm(aa)
    axis = (aa[0] / angle, aa[1] / angle, aa[2] / angle)
    angle_2 = angle / 2
    return [axis[0] * np.sin(angle_2), axis[1] * np.sin(angle_2), axis[2] * np.sin(angle_2), np.cos(angle_2)]


def callback(rawFrame):  # gets frame from realsense
    # global temporarily -- make a class later
    global tvec
    global q
    global pose_arr
    global last_q
    global last_tvec

    # capture video camera frame
    frame = bridge.imgmsg_to_cv2(rawFrame, "bgr8")

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find aruco markers in that mf image
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                     cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    if ids is not None:  # and ids[0] == id_to_find:
        # ret = [rvec, tvec, ?]
        # array of rotation and position of each marker
        # rvec = [[rvec1],[rvec2],...] attitude
        # tvec = [[tvec1].[tvec2],...] position

        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix,
                                                                           camera_distortion)
            # draw the marker and put reference frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, marker_size / 2)

            current_id = ids[i]
            if current_id == id_back:
                # adjust angle 90 degrees ccw on yaw
                # move forward 10cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0],e[1]-np.pi/2,e[2]]
                q = quaternion_from_euler(e[0],e[1],e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0]+0.10,tvec[1],tvec[2]]
            elif current_id == id_back_right:
                # adjust angle 45 degrees ccw
                # move forward 7cm, left 6 cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1]- np.pi / 4, e[2] ]
                q = quaternion_from_euler(e[0],e[1],e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] + 0.07, tvec[1] , tvec[2]- 0.06]
            elif current_id == id_right:
                # angle is good
                # move left 8 cm
                q = aa2quat(rvec[0][0])
                tvec = tvec[0][0]
                tvec = [tvec[0], tvec[1] , tvec[2]- 0.08]
            elif current_id == id_front_right_1:
                # angle rotate 30 degrees cw
                # move backward 5 cm, left 9 cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] + np.pi / 6, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0]- 0.05, tvec[1] , tvec[2]- 0.09]
            elif current_id == id_front_right_2:
                # angle rotate 67 degrees cw
                # move backward 10 cm, left 5 cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1]+ 3*np.pi / 8, e[2] ]
                q = quaternion_from_euler(e[0],e[1],e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] - 0.10, tvec[1] , tvec[2]- 0.05]
            elif current_id == id_front:
                # adjust angle 90 degrees cw
                # move backward 10cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1]+ np.pi / 2, e[2]]
                q = quaternion_from_euler(e[0],e[1],e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] - 0.10, tvec[1], tvec[2]]
            else:
                q = last_q
                tvec = last_tvec
            last_q = q
            last_tvec = tvec

        # assemble pose message
        header = HeaderMsg()
        header.frame_id = '/camera_link'
        header.stamp = rospy.Time.now()
        pose.header = header

        pose.pose.position.x = tvec[0]
        pose.pose.position.y = tvec[1]
        pose.pose.position.z = tvec[2]
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

    # display frame TODO rviz this
    pubIm.publish(bridge.cv2_to_imgmsg(frame, encoding="passthrough"))
    pubPose.publish(pose)
    # cv2.imshow("Image window", frame)

    # Added by sam
    # publish tf frame that matches pose
    br = tf.TransformBroadcaster()
    br.sendTransform(tvec, q, rospy.Time.now(), "/laser_origin", "/camera_link")


def head_track():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, callback)
    rospy.spin()


if __name__ == '__main__':
    head_track()
