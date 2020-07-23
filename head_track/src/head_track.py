#!/usr/bin/env python
# python
import time
import copy
import array

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


def aa2quat(aa):
    """
    convert from axis angle rotation to quaternion
    :param aa: orientation in axis angle notation
    returns quaternion orientation
    """
    angle = np.linalg.norm(aa)
    axis = (aa[0] / angle, aa[1] / angle, aa[2] / angle)
    angle_2 = angle / 2
    return [axis[0] * np.sin(angle_2), axis[1] * np.sin(angle_2), axis[2] * np.sin(angle_2), np.cos(angle_2)]


def reject_outliers(data_in, factor=0.8):
    """
    takes a list of data and removes outliers
    :param data_in: raw data []
    :param factor: determines how strictly outliers are removed
    returns data w/out outliers []
    """
    quant3, quant1 = np.percentile(data_in, [75, 25])
    iqr = quant3 - quant1
    iqr_sigma = iqr / 1.34896
    med_data = np.median(data_in)
    data_out = [x for x in data_in if ((x > med_data - factor * iqr_sigma) and (x < med_data + factor * iqr_sigma))]
    return data_out


def mul_quaternion(quaternion_1, quaternion_0):
    """
    used to multiply quaternions - mul one by the other to rotate
    :param quaternion_1: base quaternion []
    :param quaternion_0: quaternion that determines rotation []
    return quaternion []
    """
    a, b, c, d = quaternion_0
    e, f, g, h = quaternion_1
    coeff_1 = a * e - b * f - c * g - d * h
    coeff_i = a * f + b * e + c * h - d * g
    coeff_j = a * g - b * h + c * e + d * f
    coeff_k = a * h + b * g - c * f + d * e
    result = [coeff_1, coeff_i, coeff_j, coeff_k]
    return result


def quatWAvgMarkley(Q, weights=None):
    """
    Averages quaternions
    :param Q: (ndarray): an Mx4 ndarray of quaternions.
    :param weights: (list): an M elements list, a weight for each quaternion.
    returns single quaternion
    """
    if weights is None:
        weights = []
        i = 0
        while i < Q.shape[0]:
            weights.append(1)
            i += 1
    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    wSum = 0
    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        A += w_i * (np.outer(q, q))  # rank 1 update
        wSum += w_i
    # scale
    A /= wSum
    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]


def quaternion_median(Q, axis=3):
    """
    sorts list of quaternions based on given axis and returns median
    :param Q: (ndarray): an Mx4 ndarray of quaternions.
    :param axis: which axis to sort on
    returns median quaternion
    """
    # q_list = Q[:,3]
    sorted_arr = Q[Q[:, axis].argsort()]
    # print 'sorted q_list: ', sorted_arr
    return sorted_arr[len(Q) / 2]


def reject_quaternion_outliers(q_list, factor, axis=3):
    """
    removes outliers from a list of quaternions based on given axis
    :param q_list: (ndarray): an Mx4 ndarray of quaternions.
    :param factor: determines how strictly outliers are removed
    :param axis: which axis to based outlier removal on
    returns list of quaternions - outliers
    if all list items are 'Outliers' and removed then return None
    """
    q_list = np.array(q_list)
    # check for outliers along w - rem any outliers we can from that information
    factor = factor
    out_list = q_list[:, axis]
    quant3, quant1 = np.percentile(out_list, [75, 25])
    iqr = quant3 - quant1
    iqrSigma = iqr / 1.34896
    medData = np.median(out_list)
    i = 0
    indices_rem = []
    while i < len(out_list):
        if medData - factor * iqrSigma < out_list[i] < medData + factor * iqrSigma:
            indices_rem.append(i)
        i += 1

    q_list_filtered = q_list[indices_rem]
    if len(q_list_filtered) < 1:
        #print 'REMOVED ALL'
        return None
    else:
        # print 'len of q_list_filtered: ', len(q_list_filtered)
        #print 'REMOVED: ', (len(q_list) - len(q_list_filtered))
        #print str(len(q_list_filtered)) + '/' + str(len(q_list))
        return np.array(q_list_filtered)


def average_position(xyz_list, rej_factor, axis):
    """
    takes a list of xyz coordinates and returns the average
    :param xyz_list: list of xyz coordinates
    :param rej_factor: determines how strictly outliers are removed
    :param axis: which axis to based outlier removal on
    returns average xyz
    if all list items are 'Outliers' and removed then return median xyz
    """
    t_list = np.array(xyz_list)
    factor = rej_factor

    # check for outliers along x,y, or z - rem any outliers we can from that information
    out_list = t_list[:, axis]
    quant3, quant1 = np.percentile(out_list, [75, 25])
    iqr = quant3 - quant1
    iqrSigma = iqr / 1.34896
    medData = np.median(out_list)
    i = 0
    indices_rem = []
    while i < len(out_list):
        if medData - factor * iqrSigma < out_list[i] < medData + factor * iqrSigma:
            indices_rem.append(i)
        i += 1

    t_list_filtered = t_list[indices_rem]
    if len(t_list_filtered) < 1:
        sorted_arr = t_list[t_list[:, axis].argsort()]
        return sorted_arr[len(sorted_arr) / 2]
    else:
        return [np.mean(t_list_filtered[:, 0]), np.mean(t_list_filtered[:, 1]), np.mean(t_list_filtered[:, 2])]


def average_orientation(q_list, rej_factor=1, axis=3):
    """
    takes a list of quaternions and returns the average
    :param q_list: list of quaternions
    :param rej_factor: determines how strictly outliers are removed
    :param axis: which axis to based outlier removal on
    returns average quaternion
    if all list items are 'Outliers' and removed then return median quaternion
    """
    q_list_filtered = reject_quaternion_outliers(q_list, rej_factor, axis)
    # if all data is removed from removing outliers we take median value
    if q_list_filtered is None:
        return quaternion_median(q_list)
    else:
        return quatWAvgMarkley(q_list_filtered)


class HeadTracker:
    def __init__(self, marker_size, camera_matrix, camera_distortion, n_avg_previous_marker=10, n_avg_previous_pose=12):
        rospy.init_node('head_tracker', anonymous=True)
        # TODO: Change to subscribe to image topic published by scooter
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
        # marker orientation
        self.n_avg_previous_marker = n_avg_previous_marker
        self.marker_orient_arr = []
        def_quat = [0, 0, 0, 1]
        i = 0
        while i < self.n_avg_previous_marker:
            self.marker_orient_arr.insert(i, def_quat)
            i += 1
        # marker position
        self.marker_pos_arr = []
        def_position = [0, 0, 0]
        i = 0
        while i < self.n_avg_previous_marker:
            self.marker_pos_arr.insert(i, def_position)
            i += 1

        # number of previous poses we average with
        # larger number means smoother motion, but trails behind longer
        self.n_avg_previous_pose = n_avg_previous_pose
        self.pose_arr = []
        def_pose = geometry_msgs.msg.PoseStamped()
        header = HeaderMsg()

        # TODO: switch to base_link for scooter (already done for you)
        header.frame_id = '/base_link'
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
        while i < self.n_avg_previous_pose:
            self.pose_arr.insert(i, def_pose)
            i += 1

        self.last_tvec = [0, 0, 0]
        self.last_q = [0, 0, 0, 1]

        self.t1_4 = 0

        self.outPose = geometry_msgs.msg.PoseStamped()
        self.final_pose = def_pose

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
        q_list = []
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
                #   NOTE:   angles here are applied to y instead of z         #
                #           due to an issue on my end where everything is     #
                #           rotated 90 degrees.                               #
                # ----------------------------------------------------------- #
                # I used https://www.andre-gaschler.com/rotationconverter/ to get quaternion rot values
                current_id = ids[i]
                if current_id == self.id_back:
                    # adjust angle 90 degrees ccw
                    q_rot = [0, -.7071068, 0, 0.7071068]
                    q = aa2quat((rvec[0][0]))
                    q = mul_quaternion(q, q_rot)
                    # move forward 8.28cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0] + 0.0828, tvec[1], tvec[2]]
                elif current_id == self.id_back_R:
                    # adjust angle 45 degrees ccw
                    q_rot = [0, -0.3826834, 0, 0.9238795]
                    q = aa2quat((rvec[0][0]))
                    q = mul_quaternion(q, q_rot)
                    # move forward 5.94cm, left 5.94cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0] + 0.0594, tvec[1], tvec[2] - 0.0594]
                elif current_id == self.id_right:
                    # angle is good
                    # move left 8.28cm
                    q = aa2quat(rvec[0][0])
                    tvec = tvec[0][0]
                    tvec = [tvec[0], tvec[1], tvec[2] - 0.08]
                elif current_id == self.id_front_R:
                    # angle rotate 45 degrees cw
                    q_rot = [0, 0.3826834, 0, 0.9238795]
                    q = aa2quat((rvec[0][0]))
                    q = mul_quaternion(q, q_rot)
                    # move backward 5.94cm, left 5.94cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0] - 0.0594, tvec[1], tvec[2] - 0.0594]
                elif current_id == self.id_front:
                    # adjust angle 90 degrees cw
                    q_rot = [0, 0.7071068, 0, 0.7071068]
                    q = aa2quat((rvec[0][0]))
                    q = mul_quaternion(q, q_rot)
                    # move backward 8.28cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0] - 0.0828, tvec[1], tvec[2]]
                elif current_id == self.id_front_L:
                    # angle rotate 135 degrees cw
                    q_rot = [0, 0.9238795, 0, 0.3826834]
                    q = aa2quat((rvec[0][0]))
                    q = mul_quaternion(q, q_rot)
                    # move backward 5.94cm, right 5.94cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0] - 0.0594, tvec[1], tvec[2] + 0.0594]
                elif current_id == self.id_left:
                    # angle rotate 180 degrees cw
                    q_rot = [0, 1, 0, 0]
                    q = aa2quat(rvec[0][0])
                    q = mul_quaternion(q, q_rot)
                    # move left 8.28cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0], tvec[1], tvec[2] - 0.05]
                elif current_id == self.id_back_L:
                    # angle rotate 135 degrees ccw
                    q_rot = [0, -0.9238795, 0, 0.3826834]
                    q = aa2quat((rvec[0][0]))
                    q = mul_quaternion(q, q_rot)
                    # move forward 5.94cm, right 5.94cm
                    tvec = tvec[0][0]
                    tvec = [tvec[0] + 0.0594, tvec[1], tvec[2] + 0.0594]
                else:
                    q = self.last_q
                    tvec = self.last_tvec
                # TODO: I think this is right so leave it alone for now, but if you run into issues be sure to
                #  investigate this
                # SWAP Y and Z for Scooter
                q = [q[0], q[2], q[1], q[3]]

                self.last_q = copy.deepcopy(q)
                self.last_tvec = copy.deepcopy(tvec)

                # used for averaging
                q_list.insert(i, q)
                tvec_list_x.insert(i, tvec[0])
                tvec_list_y.insert(i, tvec[1])
                tvec_list_z.insert(i, tvec[2])
            t2 = time.time()

            # average orientation and position of all currently viewable markers
            t3 = time.time()
            # averaging orientation
            q_list = np.array(q_list)
            q = average_orientation(q_list, 0.8, 1)  # rej_factor, axis  # 1, 3
            # averaging position
            i = 0
            t_list = []
            while i < len(ids):
                t_list.append([tvec_list_x[i], tvec_list_y[i], tvec_list_z[i]])
                i += 1
            t_list = np.array(t_list)
            tvec = average_position(t_list, 100, 0)  # rej_factor, axis
            t4 = time.time()

            # average orientation and position of previous markers
            t9 = time.time()
            self.marker_orient_arr.append(copy.deepcopy(q))
            self.marker_orient_arr = self.marker_orient_arr[1:self.n_avg_previous_marker + 1]
            q_list = np.array(self.marker_orient_arr)
            q = average_orientation(q_list, 0.5, 1)

            # average position of markers with previous markers
            self.marker_pos_arr.append(copy.deepcopy(tvec))
            self.marker_pos_arr = self.marker_pos_arr[1:self.n_avg_previous_marker + 1]
            t_list = np.array(self.marker_pos_arr)
            tvec = average_position(t_list, 100, 0)
            t10 = time.time()

            # assemble pose message
            pose = geometry_msgs.msg.PoseStamped()

            header = HeaderMsg()
            # TODO: switch to base_link for scooter  (already done for you)
            header.frame_id = '/base_link'
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

            time_message = "Identifying Markers: {} seconds\nProcessing Markers: {} seconds\nAveraging Markers: {} " \
                           "seconds\nAveraging previous {} marker(s): {} seconds"
            print(time_message.format(t8 - t7, t2 - t1, t4 - t3, self.n_avg_previous_marker, t10 - t9))
            self.t1_4 = (t2 - t1 + t4 - t3 + t10 - t9)
        else:
            self.t1_4 = 0
            time_message = "Identifying Markers: {} seconds"
            print(time_message.format(t8 - t7))
            print "No Markers detected"
        # display frame
        self.pubIm.publish(bridge.cv2_to_imgmsg(frame, encoding="passthrough"))

    def average_poses(self):
        t5 = time.time()
        # add most recent pose to array of poses
        self.pose_arr.append(copy.deepcopy(self.outPose))
        self.pose_arr = self.pose_arr[1:self.n_avg_previous_pose + 1]

        # average position and orientation of poses
        q_list = []
        t_list = []
        for poses in self.pose_arr:
            t_list.append([poses.pose.position.x, poses.pose.position.y, poses.pose.position.z])
            q = [poses.pose.orientation.x, poses.pose.orientation.y, poses.pose.orientation.z, poses.pose.orientation.w]
            q_list.append(q)
        # average position
        t_list = np.array(t_list)
        tvec = average_position(t_list, 100, 0)  # rej_factor, axis
        # average orientation
        q_list = np.array(q_list)
        q = average_orientation(q_list, 1, 1)
        # end averaging pose to pose
        t6 = time.time()

        # assembling pose message
        final_pose = geometry_msgs.msg.PoseStamped()
        header = HeaderMsg()
        # TODO TODO: switch to base_link for scooter  (already done for you)
        header.frame_id = '/base_link'
        header.stamp = rospy.Time.now()
        final_pose.header = header
        final_pose.pose.position.x = tvec[0]
        final_pose.pose.position.y = tvec[1]
        final_pose.pose.position.z = tvec[2]
        final_pose.pose.orientation.x = q[0]
        final_pose.pose.orientation.y = q[1]
        final_pose.pose.orientation.z = q[2]
        final_pose.pose.orientation.w = q[3]

        time_message = "Averaging previous {} pose(s): {} seconds "
        total = "Total time: {} seconds\n"
        print(time_message.format(self.n_avg_previous_pose, t6 - t5))
        print(total.format(self.t1_4 + t6 - t5))

        self.final_pose = final_pose
        self.publish(final_pose)

    # TODO: switch to base_link for scooter  (already done for you)
    def publish(self, pose, frame="/laser_origin", parent="/base_link"):
        if pose is None:
            pose = self.final_pose
        # publish pose
        self.pubPose.publish(pose)

        # publish tf frame that matches pose
        t = [0, 0, 0]
        q = [0, 0, 0, 1]
        t[0] = pose.pose.position.x
        t[1] = pose.pose.position.y
        t[2] = pose.pose.position.z
        q[0] = pose.pose.orientation.x
        q[1] = pose.pose.orientation.y
        q[2] = pose.pose.orientation.z
        q[3] = pose.pose.orientation.w
        br = tf.TransformBroadcaster()
        # move ray down to shoot from user's eyeline
        # br.sendTransform(t, q, rospy.Time.now(), "/tracking_markers", "/camera_link")

        # move down x cm and forward x cm
        # t = [0.10,-0.22,0]
        # q = [0,0,0,1]
        # br.sendTransform(t, q, rospy.Time.now(), "/laser_origin", "/tracking_markers")
        br.sendTransform(t, q, rospy.Time.now(), frame, parent)


def head_track():
    # TODO: select your size of marker in m
    # marker_size
    # marker_size = 0.065   # 1x1
    # marker_size = 0.03    # 2x2
    marker_size = 0.02  # 3x3

    # get camera calibration
    # TODO: have a path for these on the scooter
    # TODO: have different matrix and distortion for camera used on scooter
    calib_path = '/home/csrobot/catkin_ws/src/head_track/calibration/'
    camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
    camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

    # averaging
    n_previous_marker = 8
    n_previous_pose = 10

    # Create object
    HT = HeadTracker(marker_size, camera_matrix, camera_distortion, n_previous_marker, n_previous_pose)

    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        # average across previous x poses / and publish
        HT.average_poses()
        rate.sleep()


if __name__ == '__main__':
    head_track()
