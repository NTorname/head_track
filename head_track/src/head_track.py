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

# numpy
import numpy as np

# cv
import cv2.aruco, cv2
from cv_bridge import CvBridge

# set up link between ros topics and opencv
bridge = CvBridge()


# This is mostly working, so im just going to leave it
# close enough for now.
# I plan to replace this with tf stuff i think
def move_xyz_along_axis(xyz, q_orientation, axis, distance):
    xyz = copy.deepcopy(xyz)
    q_orientation = copy.deepcopy(q_orientation)

    # notes
    # x += cos(xy_angle) yaw
    # y += sin(xy_angle) yaw
    # x += cos(xz_angle) pitch
    # z += sin(xz_angle) pitch
    # y += cos(yz_angle) roll
    # z += sin(yz_angle) roll

    if axis == "y" or axis == "Y":
        q_rot = [0, 0, 0.7071068, 0.7071068]
        q_orientation = mul_quaternion(q_orientation, q_rot)
    elif axis == "z" or axis == "Z":
        q_rot = [0, 0.7071068, 0, 0.7071068]
        q_orientation = mul_quaternion(q_orientation, q_rot)

    euler_angle = euler_from_quaternion(q_orientation)
    xy_angle = euler_angle[2]
    xz_angle = euler_angle[1]
    yz_angle = euler_angle[0]

    xyz[0] += distance * np.cos(xy_angle) if np.cos(xy_angle) < np.cos(xz_angle) else distance * np.cos(xz_angle)
    xyz[1] += distance * np.sin(xy_angle)  # if np.sin(xy_angle) < np.sin(yz_angle) else distance * np.sin(yz_angle)
    xyz[2] -= distance * np.sin(xz_angle)  # if np.sin(xz_angle) < np.cos(yz_angle) else distance * np.cos(yz_angle)
    return xyz


def aa2quat(aa):
    """
    convert from axis angle rotation to quaternion
    :param aa: orientation in axis angle notation
    returns quaternion orientation
    """
    angle = np.linalg.norm(aa)
    axis = (aa[0] / angle, aa[1] / angle, aa[2] / angle)
    angle_2 = angle / 2
    q = [axis[0] * np.sin(angle_2), axis[1] * np.sin(angle_2), axis[2] * np.sin(angle_2), np.cos(angle_2)]
    return q


def aa2quatSam(aa):
    """
    takes in axis angle in form (x,y,z,angle)
    convert from axis angle rotation to quaternion
    :param aa: orientation in axis angle notation
    returns quaternion orientation
    """
    angle = aa[3]
    factor = np.sin(angle / 2.0)
    x = aa[0] * factor
    y = aa[1] * factor
    z = aa[2] * factor
    w = np.cos(angle / 2.0)

    q = [x, y, z, w]
    # q = np.linalg.norm(q)
    # print "q: ", q
    return q


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


def norm_quat(q):
    q_x = q[0]
    q_y = q[1]
    q_z = q[2]
    q_w = q[3]
    return np.sqrt(q_x * q_x + q_y * q_y + q_z * q_z + q_w * q_w)


def is_unit_quat(q, tolerance=1e-14):
    """Determine whether the quaternion is of unit length to within a specified tolerance value.
    Params:
        tolerance: [optional] maximum absolute value by which the norm can differ from 1.0 for the object to be considered a unit quaternion. Defaults to `1e-14`.
    Returns:
        `True` if the Quaternion object is of unit length to within the specified tolerance value. `False` otherwise.
    """
    return abs(1.0 - np.dot(q, q)) < tolerance  # if np.dot(q, q) is 1, norm is 1. This saves a call to sqrt()


def normalize_quat(q):
    """
    Object is guaranteed to be a unit quaternion after calling this
    operation UNLESS the object is equivalent to Quaternion(0)
    """
    q_orig = q
    if not is_unit_quat(q):
        n = norm_quat(q)
        if n > 0:
            return q / n
    return q_orig  # return old q if nothing can be done


def fast_normalize_quat(q):
    """
    Normalise the object to a unit quaternion using a fast approximation method if appropriate.
    Object is guaranteed to be a quaternion of approximately unit length
    after calling this operation UNLESS the object is equivalent to Quaternion(0)
    """
    q_orig = q
    if not is_unit_quat(q):
        mag_squared = np.dot(q, q)
        if mag_squared == 0:
            return
        if abs(1.0 - mag_squared) < 2.107342e-08:
            mag = ((1.0 + mag_squared) / 2.0)  # More efficient. Pade approximation valid if error is small
        else:
            mag = np.sqrt(
                mag_squared)  # Error is too big, take the performance hit to calculate the square root properly

        return q / mag
    else:
        return q_orig


def slerp(q0, q1, amount=0.5):
    """Spherical Linear Interpolation between quaternions.
    Implemented as described in https://en.wikipedia.org/wiki/Slerp
    Find a valid quaternion rotation at a specified distance along the
    minor arc of a great circle passing through any two existing quaternion
    endpoints lying on the unit radius hypersphere.
    This is a class method and is called as a method of the class itself rather than on a particular instance.
    Params:
        q0: first endpoint rotation as a Quaternion object
        q1: second endpoint rotation as a Quaternion object
        amount: interpolation parameter between 0 and 1. This describes the linear placement position of
            the result along the arc between endpoints; 0 being at `q0` and 1 being at `q1`.
            Defaults to the midpoint (0.5).
    Returns:
        A new Quaternion object representing the interpolated rotation. This is guaranteed to be a unit quaternion.
    Note:
        This feature only makes sense when interpolating between unit quaternions (those lying on the unit radius hypersphere).
            Calling this method will implicitly normalise the endpoints to unit quaternions if they are not already unit length.
    """
    # Ensure quaternion inputs are unit quaternions and 0 <= amount <=1
    q0 = normalize_quat(q0)  # fast_normalize_quat(q0) change
    q1 = normalize_quat(q1)  # fast_normalize_quat(q1) change
    amount = np.clip(amount, 0, 1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negation is applied to all four components.
    # Fix by reversing one quaternion
    if dot < 0.0:
        q0 = -q0
        dot = -dot

    # sin_theta_0 can not be zero
    if dot > 0.9995:
        qr = q0 + amount * (q1 - q0)
        qr = normalize_quat(qr)  # fast_normalize_quat(qr) change
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = (s0 * q0) + (s1 * q1)
    qr = normalize_quat(qr)  # fast_normalize_quat(qr) change
    return qr


def slerp_q_list(q_list):
    """
    slerps all the quaternions to 'avg' them
    avg between first 2, take thant, avg it against next, repeat
    param q_list: list of quaternions
    returns: single quaternion
    """
    if len(q_list) > 2:
        q_avg = slerp(q_list[0], q_list[1])
        i = 2
        while i < len(q_list):
            # print 'q_avg: ', q_avg
            # print 'q_list[i]: ', q_list[i]
            q_avg = slerp(q_avg, q_list[i])
            # print 'new_q_avg: ', q_avg
            i += 1
        return q_avg
    elif len(q_list) == 2:
        q_avg = slerp(q_list[0], q_list[1])
        return q_avg
    else:
        return q_list[0]


def q_average(Q):
    """
    Averages quaternions
    :param Q: (ndarray): an Mx4 ndarray of quaternions.
    returns: single quaternion
    """
    A = np.zeros((4, 4))
    M = Q.shape[0]
    for i in range(M):
        q = Q[i, :]
        if q[3] < 0:
            q = -q
        A += np.outer(q, q)
    A /= M

    eigvals, eigvecs = np.linalg.eig(np.matmul(Q.T, Q))
    return eigvecs[:, eigvals.argmax()]


def approximately_orientation(quaternion_1, q_value, acceptable_range):
    """
    Determines if quaternion_1 is approximately equal to q_value
    param quaternion_1: quaternion
    param q_value: quaternion we compare against
    param acceptable_range: range that is acceptable
    returns: 1 if approximately equal, 0 otherwise
    """
    return 1 - (np.abs(np.dot(quaternion_1, q_value)) < acceptable_range)


def reject_quaternion_outliers(q_list, comparison_q, rej_factor):
    """
    Takes in a list of quaternions and removes quaternions that are more than
    the rej_factor from the comparison quaternion
    ex.
    In q_list quaternion 4 is .96 different to the comparison q and the
    rejection_factor is .99 . We remove quaternion 4 b/c the difference is
    less than the factor
    param q_list: list of quaternions
    param comparison_q: quaternion we compare against
    param rej_factor: rej_factor (similarity from 0 to 1)
    returns: list of quaternions with outliers removed
    """
    # [1] check if each quaternion in list is approximately equal to the
    # comparison quaternion; if so do nothing, if not, remove it
    indices_to_keep = []
    q_list = np.array(q_list)
    i = 0
    while i < len(q_list):
        if approximately_orientation(q_list[i], comparison_q, rej_factor):
            # want to keep! save the index!
            indices_to_keep.append(i)
        i += 1
    q_list = q_list[indices_to_keep]

    return q_list


def approximately_pos(pos_1, compare_v, acceptable_range):
    """
    Determines if pos_1 is approximately equal to compare_v
    param pos_1: [x,y,z] position
    param compare_v: position we compare against
    param acceptable_range: range that is acceptable
    returns: 1 if approximately equal, 0 otherwise
    """
    return np.abs(pos_1[0] - compare_v[0]) < acceptable_range and np.abs(
        pos_1[1] - compare_v[1]) < acceptable_range and np.abs(pos_1[2] - compare_v[2]) < acceptable_range


def reject_position_outliers(t_list, comparison_t, rej_factor):
    """
    Takes in a list of positions and removes positions that are more than
    the rej_factor from the comparison position
    param t_list: list of positions
    param comparison_t: position we compare against
    param rej_factor: rej_factor (similarity from 0 to 1)
    returns: list of position with outliers removed
    """
    indices_to_keep = []
    t_list = np.array(t_list)
    i = 0
    while i < len(t_list):
        if approximately_pos(t_list[i], comparison_t, rej_factor):
            # want to keep! save the index!
            indices_to_keep.append(i)
        i += 1
    # print "removed ", len(t_list)-len(indices_to_keep), "/", len(t_list), " from t_list"
    t_list = t_list[indices_to_keep]

    return t_list


class HeadTracker:
    def __init__(self, marker_size, camera_matrix, camera_distortion, parent_link, eye_height, eye_depth, image_topic,
                 n_avg_previous_marker=18):
        rospy.init_node('head_tracker', anonymous=True)
        rospy.Subscriber(image_topic, sensor_msgs.msg.Image, self.callback)

        self.parent_link = parent_link
        self.eye_height = eye_height
        self.eye_depth = eye_depth

        # set up image publisher
        self.pubIm = rospy.Publisher("/detected_frame", sensor_msgs.msg.Image, queue_size=1)
        self.pubPose = rospy.Publisher("/head_pose", geometry_msgs.msg.PoseStamped, queue_size=1)

        # get camera calibration
        self.camera_matrix = camera_matrix
        self.camera_distortion = camera_distortion

        # define aruco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.marker_size = marker_size

        # Aruco tag id's
        self.id_back = 19
        self.id_back_R = 5
        self.id_back_L = 17
        self.id_right = 1
        self.id_front_R = 4
        self.id_front = 16
        self.id_front_L = 18
        self.id_left = 10

        self.last_q = [0, 0, 0, 1]
        self.last_t = [0, 0, 0]

        # number of previous markers we average with
        # larger number means smoother motion, but trails behind longer
        self.n_avg_previous_marker = n_avg_previous_marker
        self.marker_orient_arr = []
        def_quat = [0, 0, 0, 1]
        i = 0
        while i < self.n_avg_previous_marker:
            self.marker_orient_arr.insert(i, def_quat)
            i += 1
        self.marker_pos_arr = []
        def_position = [0, 0, 0]
        i = 0
        while i < self.n_avg_previous_marker:
            self.marker_pos_arr.insert(i, def_position)
            i += 1

        self.prev_q = def_quat
        self.prev_q2 = def_quat
        self.outPose = geometry_msgs.msg.PoseStamped()

    def callback(self, raw_frame):
        t_total_first = time.time()
        # capture video camera frame
        frame = bridge.imgmsg_to_cv2(raw_frame, "bgr8")
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
        t_list = []

        t1 = time.time()
        # if markers were found
        if ids is not None:
            # loop through markers
            for i in range(0, len(ids)):
                rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.marker_size,
                                                                                self.camera_matrix,
                                                                                self.camera_distortion)
                # draw the marker and put reference frame
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.aruco.drawAxis(frame, self.camera_matrix, self.camera_distortion, rvec, tvec, self.marker_size / 2)

                # rotate and move markers depending on their position on the headset
                # I used https://www.andre-gaschler.com/rotationconverter/ to get quaternion rot values
                current_id = ids[i]
                q = aa2quat((rvec[0][0]))
                tvec = tvec[0][0]
                # tvec = [tvec[0] / 10, tvec[1] / 10, tvec[2] / 10]
                # !!!
                move_cm = -0.175 / 4
                tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                if current_id == self.id_back:
                    # adjust angle 90 degrees ccw
                    # # q_rot = [0, 0.7071068, 0, 0.7071068]
                    q_rot = aa2quatSam([0, 1, 0, -np.pi / 2])
                    q = mul_quaternion(q, q_rot)
                    q_rot = aa2quatSam([1, 0, 0, 0])
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_back_R:
                    # adjust angle 45 degrees ccw
                    q_rot = [0.9238795, 0, 0.3826834, 0]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_right:
                    # do nothing
                    pass
                    # angle is good
                elif current_id == self.id_front_R:
                    # angle rotate 45 degrees cw
                    q_rot = [0.9238795, 0, -0.3826834, 0]  # q_rot = [0, 0.3826834, 0, 0.9238795]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_front:
                    # adjust angle 90 degrees cw
                    # # q_rot = [0.7071068, 0, -0.7071068, 0]
                    q_rot = aa2quatSam([0, 1, 0, np.pi / 2])
                    q = mul_quaternion(q, q_rot)
                    q_rot = aa2quatSam([1, 0, 0, 0])
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_front_L:
                    # angle rotate 135 degrees cw
                    q_rot = [0.3826834, 0, -0.9238795, 0]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_left:
                    # angle rotate 180 degrees cw
                    q_rot = [0, 0, 1, 0]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_back_L:
                    # angle rotate 135 degrees ccw
                    q_rot = [0.3826834, 0, 0.9238795, 0]
                    q = mul_quaternion(q, q_rot)
                else:
                    # only in here if detected tag that doesn't belong to list
                    q = self.last_q
                    tvec = self.last_t
                    q_list.append(q)
                    t_list.append(tvec)
                    break

                self.last_q = q
                self.last_t = tvec

                # used for averaging
                # creating lists of xyz's and q's
                q_list.append(q)
                t_list.append(tvec)

            t2 = time.time()
            q_list = np.array(q_list)

            # average orientation and position of all currently viewable markers
            t3 = time.time()
            if len(ids) > 1:
                # [1a] remove outliers by
                # comparing list to specified quaternion
                # return list
                q_list = reject_quaternion_outliers(q_list, q_average(q_list), 0.99)

                # [1b] if all list items are removed
                # handle it
                if len(q_list) < 1:
                    print "All items removed from QUATERNION of CURRENT markers"
                    # idk, use last good q?
                    q = self.marker_orient_arr[self.n_avg_previous_marker - 1]
                else:
                    # [2] find the average position between
                    # remaining values in the list using SLERP
                    q = slerp_q_list(q_list)

                #
                # averaging position
                # [1a] remove outliers by
                # comparing list to specified xyz
                # returns list
                t_list = np.array(t_list)
                avg_xyz = [np.average(t_list[:, 0]), np.average(t_list[:, 1]), np.average(t_list[:, 2])]
                t_list = reject_position_outliers(t_list, avg_xyz, 0.03)

                # [1b] if all list items are removed
                # handle it
                if len(t_list) < 1:
                    print "All items removed from POSITION of CURRENT markers"
                    # avg all t_list
                    tvec = avg_xyz
                else:
                    # [2] find the average position between
                    # remaining values via averaging x's, y's, and z's
                    tvec = [np.average(t_list[:, 0]), np.average(t_list[:, 1]), np.average(t_list[:, 2])]
            t4 = time.time()

            # average orientation and position of previous x markers
            t9 = time.time()
            if self.n_avg_previous_marker > 1:
                # [1] append new q to list of previous orientations
                # then remove oldest
                self.marker_orient_arr.append(copy.deepcopy(q))
                self.marker_orient_arr.pop(0)  # remove oldest orientation

                # [2a] remove outliers by
                # comparing list to most recent item on list
                q_list = np.array(self.marker_orient_arr)
                q_list = reject_quaternion_outliers(q_list, self.marker_orient_arr[self.n_avg_previous_marker - 1],
                                                    0.90)

                # [2b] if all list items are removed
                # handle it
                if len(q_list) < 1:
                    print "All items removed from QUATERNION of PREVIOUS markers"
                    # use last good q?
                    q = self.marker_orient_arr[self.n_avg_previous_marker - 1]
                else:
                    # [3] ind the average position between
                    # remaining values in the list using SLERP
                    q = slerp_q_list(q_list)

                #
                # averaging position
                # [1] append new tvec to list of previous positions
                # then remove oldest
                self.marker_pos_arr.append(copy.deepcopy(tvec))
                self.marker_pos_arr.pop(0)  # remove oldest position

                # [2a] remove outliers by
                # comparing list to specified xyz
                # returns list
                t_list = np.array(self.marker_pos_arr)
                t_list_copy = t_list
                avg_xyz = [np.average(t_list[:, 0]), np.average(t_list[:, 1]), np.average(t_list[:, 2])]
                t_list = reject_position_outliers(t_list, avg_xyz, 0.01)

                # [2b] if all list items are removed
                # handle it
                if len(t_list) < 1:
                    print "All items removed from POSITION of PREVIOUS markers"
                    # avg across all
                    tvec = [np.average(t_list_copy[:, 0]), np.average(t_list_copy[:, 1]), np.average(t_list_copy[:, 2])]
                    pass
                else:
                    # [3] find the average position between
                    # remaining values via averaging x's, y's, and z's
                    tvec = [np.average(t_list[:, 0]), np.average(t_list[:, 1]), np.average(t_list[:, 2])]
            t10 = time.time()

            # rotate 90 so z is up
            q_rot = [0.7071068, 0, 0, 0.7071068]
            q = mul_quaternion(q, q_rot)

            # assemble pose message
            pose = geometry_msgs.msg.PoseStamped()
            header = HeaderMsg()
            header.frame_id = self.parent_link
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

            # print time info
            time_message = "Identifying Markers: {} seconds\nEstimating {} Poses: {} seconds\nAveraging {} Markers: " \
                           "{} seconds\nAveraging previous {} marker(s): {} seconds"
            print(time_message.format(t8 - t7, len(ids), t2 - t1, len(ids), t4 - t3, self.n_avg_previous_marker,
                                      t10 - t9))
        else:
            time_message = "Identifying Markers: {} seconds"
            print(time_message.format(t8 - t7))
            print "No Markers detected"
        # display frame
        self.pubIm.publish(bridge.cv2_to_imgmsg(frame, encoding="passthrough"))

        self.publish(self.outPose, "/tracking_markers")
        # print 'pose: ', self.outPose

        t_total_last = time.time()
        total = "Total time: {} seconds\n"
        print(total.format(t_total_last - t_total_first))
        # # test code (ignore)
        # t = move_xyz_along_axis(tvec, q, "y", 0.5)
        # br = tf.TransformBroadcaster()
        # br.sendTransform(t, q, rospy.Time.now(), "/new", self.parent_link)

    def publish(self, pose, name):
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
        #br.sendTransform(t, q, rospy.Time.now(), "/tracking_markers", self.parent_link)
        br.sendTransform(t, q, rospy.Time.now(), name, self.parent_link)

        # move ray down to shoot from user's eye-line
        # move down x cm and forward x cm
        t = [self.eye_depth, 0, self.eye_height]
        q = [0, 0, 0, 1]
        br.sendTransform(t, q, rospy.Time.now(), "/laser_origin", "/tracking_markers")
        q_rot = [0.5, 0.5, 0.5, 0.5]
        # q = mul_quaternion(q, q_rot)
        # q_rot = [-0.9961947, -0.0871557, 0, 0]
        q = mul_quaternion(q, q_rot)
        br.sendTransform(t, q, rospy.Time.now(), "/usb_cam", "/tracking_markers")


def head_track():
    # TODO: select your size of marker in m
    #  (oh yeah and multiply it by 10, idk why but it tracks better, i divide distance by 10 later)
    # marker_size
    # marker_size = 0.065   # 1x1
    marker_size = 0.03 # * 10  # 2x2
    # marker_size = 0.02  # 3x3

    # get camera calibration
    # TODO: have a path for these on the scooter
    # TODO: have different matrix and distortion for camera used on scooter
    calib_path = '/home/csrobot/catkin_ws/src/head_track/calibration/'
    camera_matrix = np.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
    camera_distortion = np.loadtxt(calib_path + 'cameraDistortion.txt', delimiter=',')

    # TODO: switch to camera tf on scooter
    # parent_link
    parent_link = "/camera_link_2"

    # TODO: customize depending on head
    # eye-position
    eye_height = -0.09
    eye_depth = 0.08

    # TODO: set image_topic correctly
    # image_topic
    image_topic = "/camera/color/image_raw"

    # smoothing level (18 seems good, but maybe lower is possible)
    # eventually when we test w/ eye-tracking this could be potentially higher
    # the less the users moves their head the less annoying the 'lag' will be
    # and having smooth stable position is important if we are using that as a basis
    # for the eye-tracking
    n_previous_marker = 7  # minimum 1

    # Create object
    HT = HeadTracker(marker_size, camera_matrix, camera_distortion, parent_link, eye_height, eye_depth, image_topic,
                     n_previous_marker)

    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    head_track()
