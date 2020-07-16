#!/usr/bin/env python


# TEST VERSION NOT FINISHED CODE


import rospy, cv2.aruco, numpy, cv2, sys, time, math, tf, sensor_msgs.msg, geometry_msgs.msg
from cv_bridge import CvBridge
from std_msgs.msg import Header as HeaderMsg
import numpy as np
import copy
import array

# define tag
id_to_find = 1
marker_size = 0.10  # m

# initalize these values here (will be in init)
tvec = [0, 0, 0]
q = [0, 0, 0, 1]
# used for averaging -- not used
# def_pose = geometry_msgs.msg.PoseStamped()
# header = HeaderMsg()
# header.frame_id = '/camera_link'
# header.stamp = 0
# def_pose.header = header
# def_pose.pose.position.x = 0
# def_pose.pose.position.y = 0
# def_pose.pose.position.z = 0
# def_pose.pose.orientation.x = 0
# def_pose.pose.orientation.y = 0
# def_pose.pose.orientation.z = 0
# def_pose.pose.orientation.w = 1
# pose_arr = []
# i = 0
# while i < 10:
#     pose_arr.insert(0, def_pose)
#     i += 1

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
    # print 'aa: ', aa
    angle = np.linalg.norm(aa)
    axis = (aa[0] / angle, aa[1] / angle, aa[2] / angle)
    angle_2 = angle / 2
    q = (axis[0] * np.sin(angle_2), axis[1] * np.sin(angle_2), axis[2] * np.sin(angle_2), np.cos(angle_2))
    return q


# used for averaging -- not used
# def reject_outliers(dataIn, factor=0.8):
#     quant3, quant1 = np.percentile(dataIn, [75, 25])
#     iqr = quant3 - quant1
#     iqrSigma = iqr / 1.34896
#     medData = np.median(dataIn)
#     dataOut = [x for x in dataIn if ((x > medData - factor * iqrSigma) and (x < medData + factor * iqrSigma))]
#     return dataOut


def callback(rawFrame):  # gets frame from realsense
    # global temporarily -- make a class later
    global tvec
    global q
    global pose_arr

    # capture video camera frame
    # cap = bridge.imgmsg_to_cv2(rawFrame, "bgr8") #desired_encoding='passthrough')
    frame = bridge.imgmsg_to_cv2(rawFrame, "bgr8")
    # set camera size
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # read camera frame
    # ret, frame = cap.read()

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find aruco markers in that mf image
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                     cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    if ids is not None and ids[0] == id_to_find:
        # ret = [rvec, tvec, ?]
        # array of rotation and position of each marker
        # rvec = [[rvec1],[rvec2],...] attitude
        # tvec = [[tvec1].[tvec2],...] position
        ret = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        # unpack output only get first
        rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]  # looks like this can be eliminated if we put the two vars into
        # the above function call, doesnt hurt us to see more markers?

        # draw the marker and put reference frame
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, marker_size / 2)

        # assemble pose message
        # rospy.logwarn("rvec: %f %f %f\ntvec %d %d %d\n", rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2])
        q = aa2quat(rvec)   # convert from axis angle to quaternion

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

    # used for averaging -- not used
    #     pose_arr.insert(0, copy.deepcopy(pose))
    #     # print 'before'
    #     # print pose_arr
    #     pose_arr = pose_arr[:-1]
    #     # print 'after'
    #     # print pose_arr
    #
    # final_pose = geometry_msgs.msg.PoseStamped()
    # header = HeaderMsg()
    # header.frame_id = '/camera_link'
    # header.stamp = rospy.Time.now()
    # final_pose.header = header
    #
    # ox = []
    # oy = []
    # oz = []
    # ow = []
    # for poses in pose_arr:
    #     print 'orix: ', poses.pose.orientation.x
    #     ox.insert(0, poses.pose.orientation.x)
    #     oy.insert(0, poses.pose.orientation.y)
    #     oz.insert(0, poses.pose.orientation.z)
    #     ow.insert(0, poses.pose.orientation.w)
    #
    # ox = reject_outliers(ox)
    # oy = reject_outliers(oy)
    # oz = reject_outliers(oz)
    # ow = reject_outliers(ow)
    #
    # print 'xoutlie: ', ox
    #
    # final_pose.pose.position.x = tvec[0]
    # final_pose.pose.position.y = tvec[1]
    # final_pose.pose.position.z = tvec[2]
    # final_pose.pose.orientation.x = np.mean(ox)
    # final_pose.pose.orientation.y = np.mean(oy)
    # final_pose.pose.orientation.z = np.mean(oz)
    # final_pose.pose.orientation.w = np.mean(ow)
    # print 'finorix: ', final_pose.pose.orientation.x
    # q[0] = final_pose.pose.orientation.x
    # q[1] = final_pose.pose.orientation.y
    # q[2] = final_pose.pose.orientation.z
    # q[3] = final_pose.pose.orientation.w

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
