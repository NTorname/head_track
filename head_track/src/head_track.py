#!/usr/bin/env python
# python
import time
import copy

# ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Header as HeaderMsg
import sensor_msgs.msg, geometry_msgs.msg
import rospy
import tf

# scipy
import numpy as np

# cv
import cv2.aruco, cv2
from cv_bridge import CvBridge

# set up link between ros topics and opencv
bridge = CvBridge()


# convert from axis angle rotation to quaternion
def aa2quat(aa):
    angle = np.linalg.norm(aa)
    axis = (aa[0] / angle, aa[1] / angle, aa[2] / angle)
    angle_2 = angle / 2
    return [axis[0] * np.sin(angle_2), axis[1] * np.sin(angle_2), axis[2] * np.sin(angle_2), np.cos(angle_2)]


# used for averaging / lower factor = more strict
def reject_outliers(dataIn, factor=0.8):
    quant3, quant1 = np.percentile(dataIn, [75, 25])
    iqr = quant3 - quant1
    iqrSigma = iqr / 1.34896
    medData = np.median(dataIn)
    dataOut = [x for x in dataIn if ((x > medData - factor * iqrSigma) and (x < medData + factor * iqrSigma))]
    return dataOut


class HeadTracker:
    def __init__(self, marker_size, camera_matrix, camera_distortion):
        rospy.init_node('head_tracker', anonymous=True)
        rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, self.callback)

        # set up image publisher
        self.pubIm = rospy.Publisher("/detected_frame", sensor_msgs.msg.Image, queue_size=10)
        self.pubPose = rospy.Publisher("/head_pose", geometry_msgs.msg.PoseStamped, queue_size=10)

        # get camera calibration
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

        # define aruco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.marker_size = marker_size

        # Aruco tag id's
        self.id_back = 380
        self.id_back_R = 403
        self.id_back_L = 473
        self.id_right = 643
        self.id_front_R = 303
        self.id_front = 688
        self.id_front_L = 891
        self.id_left = 667

        # number of previous markers we average with
        # larger number means smoother motion, but trails behind longer
        self.n_avg_marker = 10
        self.marker_arr = []
        def_eul = [0,0,0]
        i = 0
        while i < self.n_avg_marker:
            self.marker_arr.insert(i, def_eul)
            i += 1

        # number of previous poses we average with
        # larger number means smoother motion, but trails behind longer
        self.n_avg_pose = 25
        self.pose_arr = []
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
        i = 0
        while i < self.n_avg_pose:
            self.pose_arr.insert(i, def_pose)
            i += 1

        self.last_tvec = [0, 0, 0]
        self.last_q = [0, 0, 0, 1]

        self.t1_4 = 0

        self.outMarker = def_eul
        self.outPose = geometry_msgs.msg.PoseStamped()

    def callback(self, rawFrame):
        # capture video camera frame
        frame = bridge.imgmsg_to_cv2(rawFrame, "bgr8")
        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # find aruco markers in that mf image
        t7 = time.time()
        corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=self.aruco_dict,
                                                         parameters=self.parameters,
                                                         cameraMatrix=self.camera_matrix,
                                                         distCoeff=self.camera_distortion)
        t8 = time.time()
        # used for averaging
        e_list_x = []
        e_list_y = []
        e_list_z = []
        tvec_list_x = []
        tvec_list_y = []
        tvec_list_z = []

        t1 = time.time()

        # if markers were found
        if ids is not None:
            # loop through markers
            for i in range(0, len(ids)):
                rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.marker_size,
                                                                                self.camera_matrix,
                                                                                self.camera_distortion)
                # draw the marker and put reference frame
                cv2.aruco.drawDetectedMarkers(frame, corners)  # , ids)
                cv2.aruco.drawAxis(frame, self.camera_matrix, self.camera_distortion, rvec, tvec, self.marker_size / 2)
                # ----------------------------------------------------------- #
                #   NOTE:   angles here are applied to pitch instead of yaw   #
                #           due to an issue on my end where everything is     #
                #           rotated 90 degrees. in the final version all      #
                #           rotations will be applied to yaw                  #
                #           (essentially mine is RYP instead of RPY)          #
                # ----------------------------------------------------------- #
                # TODO switch y and z axes for scooter implementation
                current_id = ids[i]
                if current_id == self.id_back:
                    # adjust angle 90 degrees ccw on yaw
                    # move forward 8.28cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] - np.pi / 2, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0] + 0.0828, tvec[1], tvec[2]]
                elif current_id == self.id_back_R:
                    # adjust angle 45 degrees ccw
                    # move forward 5.94cm, left 5.94cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] - np.pi / 4, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0] + 0.0594, tvec[1], tvec[2] - 0.0594]
                elif current_id == self.id_right:
                    # angle is good
                    # move left 8.28cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    tvec = tvec[0][0]
                    tvec = [tvec[0], tvec[1], tvec[2] - 0.08]
                elif current_id == self.id_front_R:
                    # angle rotate 45 degrees cw
                    # move backward 5.94cm, left 5.94cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] + np.pi / 4, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0] - 0.0594, tvec[1], tvec[2] - 0.0594]
                elif current_id == self.id_front:
                    # adjust angle 90 degrees cw
                    # move backward 8.28cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] + np.pi / 2, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0] - 0.0828, tvec[1], tvec[2]]
                elif current_id == self.id_front_L:
                    # angle rotate 135 degrees ccw
                    # move backward 5.94cm, right 5.94cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] + 3 * np.pi / 4, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0] - 0.0594, tvec[1], tvec[2] + 0.0594]
                elif current_id == self.id_left:
                    # angle rotate 180 degrees cw
                    # move left 8.28cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] + np.pi, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0], tvec[1], tvec[2] - 0.05]
                elif current_id == self.id_back_L:
                    # angle rotate 135 degrees cw
                    # move forward 5.94cm, right 5.94cm
                    q = aa2quat(rvec[0][0])
                    e = euler_from_quaternion(q)
                    e = [e[0], e[1] - 3 * np.pi / 4, e[2]]
                    tvec = tvec[0][0]
                    tvec = [tvec[0] + 0.0594, tvec[1], tvec[2] + 0.0594]
                else:
                    q = self.last_q
                    e = euler_from_quaternion(q)
                    tvec = self.last_tvec

                # used for averaging
                e_list_x.insert(i, e[0])
                e_list_y.insert(i, e[1])
                e_list_z.insert(i, e[2])
                tvec_list_x.insert(i, tvec[0])
                tvec_list_y.insert(i, tvec[1])
                tvec_list_z.insert(i, tvec[2])

            t2 = time.time()
            t3 = time.time()

            # averaging pose between all markers
            q_rej_rate = 0.8   # 0.40
            tvec_rej_rate = 100  # 0.80
            e_list_x_filtered = reject_outliers(e_list_x, q_rej_rate)
            e_list_y_filtered = reject_outliers(e_list_y, q_rej_rate)
            e_list_z_filtered = reject_outliers(e_list_z, q_rej_rate)
            tvec_list_x_filtered = reject_outliers(tvec_list_x, tvec_rej_rate)
            tvec_list_y_filtered = reject_outliers(tvec_list_y, tvec_rej_rate)
            tvec_list_z_filtered = reject_outliers(tvec_list_z, tvec_rej_rate)

            # used for averaging pose between all markers
            # if reject outliers phase removes all marker quaternion we use last useful quaternion
            if len(e_list_x_filtered) < 1 or len(e_list_y_filtered) < 1 or len(e_list_z_filtered) < 1:
                # q = self.last_q
                q = quaternion_from_euler(np.mean(e_list_x), np.mean(e_list_y), np.mean(e_list_z))
            else:
                q = quaternion_from_euler(np.mean(e_list_x_filtered), np.mean(e_list_y_filtered),
                                          np.mean(e_list_z_filtered))
            # if reject outliers phase removes all marker translation we use last useful translation
            if len(tvec_list_x_filtered) < 1 or len(tvec_list_y_filtered) < 1 or len(tvec_list_z_filtered) < 1:
                # tvec = self.last_tvec
                tvec = [np.mean(tvec_list_x), np.mean(tvec_list_y), np.mean(tvec_list_z)]
            else:
                tvec = [np.mean(tvec_list_x_filtered), np.mean(tvec_list_y_filtered), np.mean(tvec_list_z_filtered)]

            self.last_q = copy.deepcopy(q)
            self.last_tvec = copy.deepcopy(tvec)
            t4 = time.time()

            # average markers with previous markers
            t9 = time.time()
            self.outMarker = euler_from_quaternion(q)
            eul = self.average_markers()
            q = quaternion_from_euler(eul[0],eul[1],eul[2])
            t10 = time.time()

            # assemble pose message
            pose = geometry_msgs.msg.PoseStamped()

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

            self.outPose = pose

            # display frame
            self.pubIm.publish(bridge.cv2_to_imgmsg(frame, encoding="passthrough"))

            time_message = "Identifying Markers: {} seconds\nProcessing Markers: {} seconds\nAveraging Markers: {} " \
                           "seconds\nAveraging previous {} marker(s): {} seconds"
            print(time_message.format(t8 - t7, t2 - t1, t4 - t3, self.n_avg_marker, t10-t9))
            self.t1_4 = (t2 - t1 + t4 - t3 + t10-t9)
        else:
            self.t1_4 = 0
            time_message = "Identifying Markers: {} seconds"
            print(time_message.format(t8 - t7))
            print "No Markers detected"

    def average_markers(self, n_avg_marker=10):
        self.n_avg_marker = n_avg_marker
        self.marker_arr.append(copy.deepcopy(self.outMarker))
        self.marker_arr = self.marker_arr[1:self.n_avg_marker + 1]

        final_marker = [0,0,0]

        ex_list = []
        ey_list = []
        ez_list = []
        for markers in self.marker_arr:
            ex_list.append(markers[0])
            ey_list.append(markers[1])
            ez_list.append(markers[2])

        rej_rate = 1.5
        ex_filtered = reject_outliers(ex_list, rej_rate)
        ey_filtered = reject_outliers(ey_list, rej_rate)
        ez_filtered = reject_outliers(ez_list, rej_rate)

        # if reject outliers phase removes all orientation we use average across non filtered orientation
        if len(ex_filtered) < 1 or len(ey_filtered) < 1 or len(ez_filtered) < 1:
            final_marker[0] = np.mean(ex_list)
            final_marker[1] = np.mean(ey_list)
            final_marker[2] = np.mean(ey_list)
        else:
            final_marker[0] = np.mean(ex_filtered)
            final_marker[1] = np.mean(ey_filtered)
            final_marker[2] = np.mean(ez_filtered)

        return final_marker

    def average_poses(self, n_avg_pose=12):
        t5 = time.time()
        self.n_avg_pose = n_avg_pose
        # averaging pose to pose
        self.pose_arr.append(copy.deepcopy(self.outPose))
        self.pose_arr = self.pose_arr[1:self.n_avg_pose + 1]

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
        for poses in self.pose_arr:
            x_list.append(poses.pose.position.x)
            y_list.append(poses.pose.position.y)
            z_list.append(poses.pose.position.z)
            eul = euler_from_quaternion(
                [poses.pose.orientation.x, poses.pose.orientation.y, poses.pose.orientation.z,
                 poses.pose.orientation.w])
            er_list.append(eul[0])
            ep_list.append(eul[1])
            ey_list.append(eul[2])

        rej_out_rate_trans = 100  # 0.75
        x_filtered = reject_outliers(x_list, rej_out_rate_trans)
        y_filtered = reject_outliers(y_list, rej_out_rate_trans)
        z_filtered = reject_outliers(z_list, rej_out_rate_trans)

        # if reject outliers phase removes all trans we use average across non filtered trans
        if len(x_filtered) < 1 or len(y_filtered) < 1 or len(z_filtered) < 1:
            final_pose.pose.position.x = np.mean(x_list)
            final_pose.pose.position.y = np.mean(y_list)
            final_pose.pose.position.z = np.mean(z_list)
        else:
            final_pose.pose.position.x = np.mean(x_filtered)
            final_pose.pose.position.y = np.mean(y_filtered)
            final_pose.pose.position.z = np.mean(z_filtered)

        rej_out_rate_orient = 1.5  # 0.75
        er_filtered = reject_outliers(er_list, rej_out_rate_orient)
        ep_filtered = reject_outliers(ep_list, rej_out_rate_orient)
        ey_filtered = reject_outliers(ey_list, rej_out_rate_orient)
        # if reject outliers phase removes all orientation we use average across non filtered orientation
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

        # TODO switch y and z axes for scooter implementation
        # move pose down so ray extends from user's eye-line
        #
        t = [0, 0, 0]
        q = [0, 0, 0, 1]
        t[0] = final_pose.pose.position.x
        t[1] = final_pose.pose.position.y
        t[2] = final_pose.pose.position.z
        q[0] = final_pose.pose.orientation.x
        q[1] = final_pose.pose.orientation.y
        q[2] = final_pose.pose.orientation.z
        q[3] = final_pose.pose.orientation.w
        # end averaging pose to pose
        t6 = time.time()

        time_message = "Averaging previous {} pose(s): {} seconds "
        total = "Total time: {} seconds\n"
        print(time_message.format(self.n_avg_pose, t6 - t5))
        print(total.format(self.t1_4 + t6 - t5))

        # publish pose
        self.pubPose.publish(final_pose)

        # publish tf frame that matches pose
        t = [0, 0, 0]
        q = [0, 0, 0, 1]
        t[0] = final_pose.pose.position.x
        t[1] = final_pose.pose.position.y
        t[2] = final_pose.pose.position.z
        q[0] = final_pose.pose.orientation.x
        q[1] = final_pose.pose.orientation.y
        q[2] = final_pose.pose.orientation.z
        q[3] = final_pose.pose.orientation.w
        br = tf.TransformBroadcaster()
        # br.sendTransform(t, q, rospy.Time.now(), "/tracking_markers", "/camera_link")

        # move down x cm and forward x cm
        # t = [0.10,-0.22,0]
        # q = [0,0,0,1]
        # br.sendTransform(t, q, rospy.Time.now(), "/laser_origin", "/tracking_markers")
        br.sendTransform(t, q, rospy.Time.now(), "/laser_origin", "/camera_link")

    # # not needed. turns out tf is great
    # def move_xyz_along_axis(self, xyz, orientation, axis, distance):
    #     if axis == "X" or axis =='x':
    #         # depends on pitch & yaw
    #         if np.cos(orientation[1]) < np.cos(orientation[2]):
    #             xyz[0] += distance * np.cos(orientation[1])
    #         else:
    #             xyz[0] += distance * np.cos(orientation[2])
    #         # depends on yaw
    #         xyz[1] += distance * np.sin(orientation[2])
    #         # depends on pitch
    #         xyz[2] -= distance * np.sin(orientation[1])
    #         return xyz
    #     elif axis == "Y" or axis =='y':
    #         # depends on yaw
    #         xyz[0] += distance * np.sin(orientation[2])
    #         # depends on yaw & roll
    #         if np.cos(orientation[0]) < np.cos(orientation[2]):
    #             xyz[1] += distance * np.cos(orientation[0])
    #         else:
    #             xyz[1] += distance * np.cos(orientation[2])
    #         # depends on roll
    #         xyz[2] += distance * np.sin(orientation[0])
    #         return xyz
    #     elif axis == "Z" or axis =='z':
    #         # depends on pitch
    #         xyz[0] += distance * np.sin(orientation[1])
    #         # depends on roll
    #         xyz[1] += distance * np.sin(orientation[0])
    #         # depends on roll & pitch
    #         if np.cos(orientation[0]) < np.cos(orientation[1]):
    #             xyz[2] += distance * np.cos(orientation[0])
    #         else:
    #             xyz[2] += distance * np.cos(orientation[1])
    #         return xyz
    #     else:
    #         return xyz


def head_track():
    # marker_size
    # 0.065     # 1x1
    marker_size = 0.03  # 4x4
    # 0.02      # 4x4

    # get camera calibration
    calib_path = '/home/csrobot/catkin_ws/src/head_track/calibration/'
    camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
    camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

    # Create object
    HT = HeadTracker(marker_size, camera_matrix, camera_distortion)

    rate = rospy.Rate(30.0)
    while not rospy.is_shutdown():
        # average across previous x poses / and publish
        HT.average_poses(12)
        rate.sleep()


if __name__ == '__main__':
    head_track()
