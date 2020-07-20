#!/usr/bin/env python


# TEST VERSION NOT FINISHED CODE


import rospy, cv2.aruco, numpy, cv2, sys, time, math, tf, sensor_msgs.msg, geometry_msgs.msg
from cv_bridge import CvBridge
from std_msgs.msg import Header as HeaderMsg
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import copy
import time
import array

# used for averaging from pose to pose
def_pose = geometry_msgs.msg.PoseStamped()
header = HeaderMsg()
header.frame_id = '/camera_link'
header.stamp = 0
def_pose.header = header
def_pose.pose.position.x = 0
def_pose.pose.position.y = 0
def_pose.pose.position.z = 0
def_pose.pose.orientation.x = 0
def_pose.pose.orientation.y = 0
def_pose.pose.orientation.z = 0
def_pose.pose.orientation.w = 1
pose_arr = []
i = 0
# number of previous poses we average with
# larger number means smoother motion, but trails behind longer
n_avg_pose = 25
while i < n_avg_pose:
    pose_arr.insert(i, def_pose)
    i += 1

# define tag
id_to_find = 1
# marker_size = 0.065  # m
marker_size = 0.02  # small markers
# Aruco tag id's
id_back = 380
id_back_R = 403
id_back_L = 473
id_right = 643
id_front_R = 303
id_front = 688
id_front_L = 891
id_left = 667

# (temp) initialize these values here (will be in init)
tvec = [0, 0, 0]
q = [0, 0, 0, 1]
last_q = [0, 0, 0, 1]
last_tvec = [0, 0, 0]

# get camera calibration
calib_path = '/home/csrobot/catkin_ws/src/head_track/calibration/'
camera_matrix = numpy.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
camera_distortion = numpy.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

# # 180 degree matrix rotation around x axis   may not be necessary for this application
# R_flip = numpy.zeros((3, 3), dtype=numpy.float32)
# R_flip[0, 0] = 1.0
# R_flip[1, 1] = -1.0
# R_flip[2, 2] = -1.0

# TODO: find way to use other aruco dictionaries
# define aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create()

# set up link between ros topics and opencv
bridge = CvBridge()

# set up image publisher
pubIm = rospy.Publisher("/detected_frame", sensor_msgs.msg.Image, queue_size=10)
pubPose = rospy.Publisher("/head_pose", geometry_msgs.msg.PoseStamped, queue_size=10)

pose = geometry_msgs.msg.PoseStamped()


# convert from axis angle rotation to quaternion
def aa2quat(aa):
    angle = np.linalg.norm(aa)
    axis = (aa[0] / angle, aa[1] / angle, aa[2] / angle)
    angle_2 = angle / 2
    return [axis[0] * np.sin(angle_2), axis[1] * np.sin(angle_2), axis[2] * np.sin(angle_2), np.cos(angle_2)]


# used for averaging
# lower factor = more strict
def reject_outliers(dataIn, factor=0.8):
    quant3, quant1 = np.percentile(dataIn, [75, 25])
    iqr = quant3 - quant1
    iqrSigma = iqr / 1.34896
    medData = np.median(dataIn)
    dataOut = [x for x in dataIn if ((x > medData - factor * iqrSigma) and (x < medData + factor * iqrSigma))]
    return dataOut


def callback(rawFrame):  # gets frame from realsense
    # global temporarily -- make a class later
    global tvec
    global q
    global pose_arr
    global last_q
    global last_tvec
    global n_avg_pose

    # capture video camera frame
    frame = bridge.imgmsg_to_cv2(rawFrame, "bgr8")

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find aruco markers in that mf image
    corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                     cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    # used for averaging
    e_list_x = []
    e_list_y = []
    e_list_z = []
    tvec_list_x = []
    tvec_list_y = []
    tvec_list_z = []

    t1 = time.time()

    if ids is not None:
        # ret = [rvec, tvec, ?]
        # array of rotation and position of each marker
        # rvec = [[rvec1],[rvec2],...] attitude
        # tvec = [[tvec1].[tvec2],...] position

        for i in range(0, len(ids)):
            rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix,
                                                                            camera_distortion)
            # draw the marker and put reference frame
            cv2.aruco.drawDetectedMarkers(frame, corners)  # , ids)
            cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, marker_size / 2)

            # ----------------------------------------------------------- #
            #   NOTE:   angles here are applied to pitch instead of yaw   #
            #           due to an issue on my end where everything is     #
            #           rotated 90 degrees. in the final version all      #
            #           rotations will be applied to yaw                  #
            # ----------------------------------------------------------- #
            current_id = ids[i]
            if current_id == id_back:
                # adjust angle 90 degrees ccw on yaw
                # move forward 8.28cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] - np.pi / 2, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] + 0.0828, tvec[1], tvec[2]]
            elif current_id == id_back_R:
                # adjust angle 45 degrees ccw
                # move forward 5.94cm, left 5.94cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] - np.pi / 4, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] + 0.0594, tvec[1], tvec[2] - 0.0594]
            elif current_id == id_right:
                # angle is good
                # move left 8.28cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                tvec = tvec[0][0]
                tvec = [tvec[0], tvec[1], tvec[2] - 0.08]
            elif current_id == id_front_R:
                # angle rotate 45 degrees cw
                # move backward 5.94cm, left 5.94cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] + np.pi / 4, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] - 0.0594, tvec[1], tvec[2] - 0.0594]
            elif current_id == id_front:
                # adjust angle 90 degrees cw
                # move backward 8.28cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] + np.pi / 2, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] - 0.0828, tvec[1], tvec[2]]
            elif current_id == id_front_L:
                # angle rotate 135 degrees ccw
                # move backward 5.94cm, right 5.94cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] + 3 * np.pi / 4, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] - 0.0594, tvec[1], tvec[2] + 0.0594]
            elif current_id == id_left:
                # angle rotate 180 degrees cw
                # move left 8.28cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] + np.pi, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0], tvec[1], tvec[2] - 0.05]
            elif current_id == id_back_L:
                # angle rotate 135 degrees cw
                # move forward 5.94cm, right 5.94cm
                q = aa2quat(rvec[0][0])
                e = euler_from_quaternion(q)
                e = [e[0], e[1] - 3 * np.pi / 4, e[2]]
                q = quaternion_from_euler(e[0], e[1], e[2])
                tvec = tvec[0][0]
                tvec = [tvec[0] + 0.0594, tvec[1], tvec[2] + 0.0594]
            else:
                q = last_q
                e = euler_from_quaternion(q)
                tvec = last_tvec

            # used for averaging
            e_list_x.insert(i, e[0])
            e_list_y.insert(i, e[1])
            e_list_z.insert(i, e[2])
            tvec_list_x.insert(i, tvec[0])
            tvec_list_y.insert(i, tvec[1])
            tvec_list_z.insert(i, tvec[2])

        t2 = time.time()
        t3 = time.time()

        # used for averaging pose between all markers
        q_rej_rate = 0.40
        tvec_rej_rate = 0.80
        e_list_x = reject_outliers(e_list_x, q_rej_rate)
        e_list_y = reject_outliers(e_list_y, q_rej_rate)
        e_list_z = reject_outliers(e_list_z, q_rej_rate)
        tvec_list_x = reject_outliers(tvec_list_x, tvec_rej_rate)
        tvec_list_y = reject_outliers(tvec_list_y, tvec_rej_rate)
        tvec_list_z = reject_outliers(tvec_list_z, tvec_rej_rate)
        # print 'tvec_list_x: ', tvec_list_x
        # print 'tvec_list_y: ', tvec_list_y
        # print 'tvec_list_z: ', tvec_list_z

        # used for averaging pose between all markers
        if len(e_list_x) < 1 or len(e_list_y) < 1 or len(e_list_z) < 1:
            # print 'last q: ', last_q
            q = last_q
        else:
            # print 'x: ', e_list_x
            # print 'y: ', e_list_y
            # print 'z: ', e_list_z
            q = quaternion_from_euler(np.mean(e_list_x), np.mean(e_list_y), np.mean(e_list_z))
        # print 'q: ', q
        if len(tvec_list_x) < 1 or len(tvec_list_y) < 1 or len(tvec_list_z) < 1:
            tvec = last_tvec
        else:
            tvec = [np.mean(tvec_list_x), np.mean(tvec_list_y), np.mean(tvec_list_z)]

        last_q = copy.deepcopy(q)
        last_tvec = tvec

        t4 = time.time()

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

        t5 = time.time()
        #
        # averaging pose to pose
        pose_arr.append(copy.deepcopy(pose))
        pose_arr = pose_arr[1:11]

        final_pose = geometry_msgs.msg.PoseStamped()
        header = HeaderMsg()
        header.frame_id = '/camera_link'
        header.stamp = rospy.Time.now()
        final_pose.header = header

        x_list = []
        y_list = []
        z_list = []
        er_list = []
        ep_list = []
        ey_list = []
        for poses in pose_arr:
            x_list.append(poses.pose.position.x)
            y_list.append(poses.pose.position.y)
            z_list.append(poses.pose.position.z)
            eul = euler_from_quaternion([poses.pose.orientation.x, poses.pose.orientation.y, poses.pose.orientation.z,poses.pose.orientation.w])
            er_list.append(eul[0])
            ep_list.append(eul[1])
            ey_list.append(eul[2])

        rej_out_rate_trans = 0.75
        x_filtered = reject_outliers(x_list, rej_out_rate_trans)
        y_filtered = reject_outliers(y_list, rej_out_rate_trans)
        z_filtered = reject_outliers(z_list, rej_out_rate_trans)
        rej_out_rate_orient = 0.75
        er_filtered = reject_outliers(er_list, rej_out_rate_orient)
        ep_filtered = reject_outliers(ep_list, rej_out_rate_orient)
        ey_filtered = reject_outliers(ey_list, rej_out_rate_orient)

        if len(x_filtered) < 1 or len(y_filtered) < 1 or len(z_filtered) < 1:
            final_pose.pose.position.x = np.mean(x_list)
            final_pose.pose.position.y = np.mean(y_list)
            final_pose.pose.position.z = np.mean(z_list)
        else:
            final_pose.pose.position.x = np.mean(x_filtered)
            final_pose.pose.position.y = np.mean(y_filtered)
            final_pose.pose.position.z = np.mean(z_filtered)
            
        if len(er_filtered) < 1 or len(ep_filtered) < 1 or len(ey_filtered) < 1:
            quat = quaternion_from_euler(np.mean(er_list), np.mean(ep_list), np.mean(ey_list))
            final_pose.pose.orientation.x = quat[0]
            final_pose.pose.orientation.y = quat[1]
            final_pose.pose.orientation.z = quat[2]
            final_pose.pose.orientation.w = quat[3]
        else:
            quat = quaternion_from_euler(np.mean(er_filtered), np.mean(ep_filtered), np.mean(ey_filtered))
            final_pose.pose.orientation.x = quat[0]
            final_pose.pose.orientation.y = quat[1]
            final_pose.pose.orientation.z = quat[2]
            final_pose.pose.orientation.w = quat[3]

        tvec[0] = final_pose.pose.position.x
        tvec[1] = final_pose.pose.position.y
        tvec[2] = final_pose.pose.position.z
        q[0] = final_pose.pose.orientation.x
        q[1] = final_pose.pose.orientation.y
        q[2] = final_pose.pose.orientation.z
        q[3] = final_pose.pose.orientation.w
        # end averaging pose to pose

        t6 = time.time()

        time_message = "Aruco stuff: {} seconds\nAveraging between Markers: {} seconds\nAveraging between previous poses: {} seconds"
        total = "Total time: {} seconds"
        print(time_message.format(t2-t1,t4-t3,t6-t5))
        print(total.format(t2-t1 + t4-t3 + t6-t5))

        # display frame
        pubIm.publish(bridge.cv2_to_imgmsg(frame, encoding="passthrough"))
        # publish pose
        pubPose.publish(final_pose)
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
