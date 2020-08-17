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

test_val = 0


# This is mostly working, so im just going to leave it
# close enough for now.
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
    elif axis == "z" or axis == "z":
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
    # print "q: ", q
    return q


def aa2quatSam(aa):
    """
    takes ib axis angle in form (x,y,z,angle)
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


def reject_outliers(data_in, factor=0.8 * test_val):
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


def conj_quat(q):
    return [-q[0], -q[1], -q[2], q[3]]


def norm_quat(q):
    q_x = q[0]
    q_y = q[1]
    q_z = q[2]
    q_w = q[3]
    return np.sqrt(q_x * q_x + q_y * q_y + q_z * q_z + q_w * q_w)


def sum_of_squares(q):
    return np.dot(q, q)


def is_unit_quat(q, tolerance=1e-14):
    """Determine whether the quaternion is of unit length to within a specified tolerance value.
    Params:
        tolerance: [optional] maximum absolute value by which the norm can differ from 1.0 for the object to be considered a unit quaternion. Defaults to `1e-14`.
    Returns:
        `True` if the Quaternion object is of unit length to within the specified tolerance value. `False` otherwise.
    """
    return abs(1.0 - sum_of_squares(q)) < tolerance  # if _sum_of_squares is 1, norm is 1. This saves a call to sqrt()


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


def inverse_quat(q):
    q_c = conj_quat(q)
    q_conj_x = q_c[0]
    q_conj_y = q_c[1]
    q_conj_z = q_c[2]
    q_conj_w = q_c[3]

    q_norm = norm_quat(q)

    q_inv_x = q_conj_x / (q_norm * q_norm)
    q_inv_y = q_conj_y / (q_norm * q_norm)
    q_inv_z = q_conj_z / (q_norm * q_norm)
    q_inv_w = q_conj_w / (q_norm * q_norm)

    return [q_inv_x, q_inv_y, q_inv_z, q_inv_w]


def divide_quat(q1, q2):
    return mul_quaternion(q1, inverse_quat(q2))


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
    q0 = fast_normalize_quat(q0)
    q1 = fast_normalize_quat(q1)
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
        qr = fast_normalize_quat(qr)
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = (s0 * q0) + (s1 * q1)
    qr = fast_normalize_quat(qr)
    return qr


def avg_quat_sam(q_list):
    # slerp 'avgs' between 2 q's
    # avg between first 2, take thant, avg it aginast next, repeat
    if len(q_list) > 2:
        q_avg = slerp(q_list[0], q_list[1])
        i = 2
        while i < len(q_list):
            q_avg = slerp(q_avg, q_list[i])
            i += 1
        return q_avg
    elif len(q_list) == 2:
        q_avg = slerp(q_list[0], q_list[1])
        return q_avg
    else:
        return q_list[0]


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


# def quaternion_median(Q, axis=3):
#     """
#     sorts list of quaternions based on given axis and returns median
#     :param Q: (ndarray): an Mx4 ndarray of quaternions.
#     :param axis: which axis to sort on
#     returns median quaternion
#     """
#     # q_list = Q[:,3]
#     sorted_arr = Q[Q[:, axis].argsort()]
#     # print 'sorted q_list: ', sorted_arr
#     return sorted_arr[len(Q) / 2]

def quaternion_median(Q, axis=3):
    # add xyzw together
    # sort that way?

    sort_arr = []
    sum = 0
    i = 0
    while i < len(Q):
        # go through all q_list
        sum += (np.abs(Q[i][0]) * np.abs(Q[i][1]) * np.abs(Q[i][2]) * np.abs(Q[i][3]))
        sort_arr.insert(i, sum)
        sum = 0
        i += 1
    # sort q_list by sort_arr?
    # print "Q: ", Q
    sort_arr = np.array(sort_arr)
    # print "Sort: ", sort_arr
    sorted_arr = Q[sort_arr.argsort()]
    # print "med_q: ", sorted_arr

    return sorted_arr[len(Q) / 2]


def reject_quaternion_outliers(q_list, axis_factor, angle_factor):
    """
    removes outliers from a list of quaternions based similarity of products
    """
    # x y z are axis
    # w is essentailly angle
    # should compare angle on each axis?

    avg_q = avg_quat_sam(q_list)

    # check x y z axis to see if they are close
    # then check if angle is close?

    axis = 0
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list):
        dif_q = q_list[i][axis] - avg_q[axis]
        # print 'q: ', q_list[i]
        # print 'median_q: ', median_q
        print 'dif_qx: ', dif_q
        if np.abs(dif_q) < axis_factor:
            # print "^^^"
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list[indices_rem]
    # print 'q_list_ filtered: ', q_list_filtered

    axis = 1
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list_filtered):
        dif_q = q_list_filtered[i][axis] - avg_q[axis]
        # print 'q: ', q_list[i]
        # print 'median_q: ', median_q
        print 'dif_qy: ', dif_q
        print 'axis_factor=',axis_factor
        if np.abs(dif_q) < axis_factor:
            # print "^^^"
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list_filtered[indices_rem]
    # print 'q_list_ filtered: ', q_list_filtered

    axis = 2
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list_filtered):
        dif_q = q_list_filtered[i][axis] - avg_q[axis]
        # print 'q: ', q_list[i]
        # print 'median_q: ', median_q
        print 'dif_qz: ', dif_q
        if np.abs(dif_q) < axis_factor:
            # print "^^^"
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list_filtered[indices_rem]
    # print 'q_list_ filtered: ', q_list_filtered

    axis = 3
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list_filtered):
        dif_q = q_list_filtered[i][axis] - avg_q[axis]
        # print 'q: ', q_list[i]
        # print 'median_q: ', median_q
        print 'dif_qw: ', dif_q
        if np.abs(dif_q) < angle_factor:
            # print "^^^"
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list_filtered[indices_rem]
    # print 'q_list_ filtered: ', q_list_filtered

    # print 'q_list: ', q_list
    # print 'q_list_filtered: ', q_list_filtered

    if len(q_list_filtered) < 1:
        # print 'REMOVED ALL (in function reject_quaternion_outliers)'
        return None
    else:
        # print 'len of q_list_filtered: ', len(q_list_filtered)
        # print 'REMOVED: ', (len(q_list) - len(q_list_filtered))
        print str(len(q_list_filtered)) + '/' + str(len(q_list))
        return np.array(q_list_filtered)


def average_position(xyz_list, rej_factor):
    # look at x, rem some from list
    # look at y, same
    # look at z, same
    # return avg of remaining

    t_list = np.array(xyz_list)

    avg_xyz = [np.mean(xyz_list[:, 0]), np.mean(xyz_list[:, 1]), np.mean(xyz_list[:, 2])]
    # print 't_list: ', t_list

    axis = 0
    # if avg x - current x < rej then keep
    indices_rem = []
    i = 0
    while i < len(t_list):
        dif_t = np.abs(avg_xyz[axis] - t_list[i][axis])
        # print 'x: ', t_list[i][axis]
        # print 'avgx: ', avg_xyz[axis]
        # print 'dif_tx: ', dif_t
        if dif_t < rej_factor:
            indices_rem.append(i)
        i += 1
    t_list_filtered = t_list[indices_rem]
    # print 't_list_filtered  : ', t_list_filtered

    axis = 1
    indices_rem = []
    i = 0
    while i < len(t_list_filtered):
        dif_t = np.abs(avg_xyz[axis] - t_list[i][axis])
        # print 'x: ', t_list[i][axis]
        # print 'avgx: ', avg_xyz[axis]
        # print 'dif_ty: ', dif_t
        if dif_t < rej_factor:
            indices_rem.append(i)
        i += 1
    t_list_filtered = t_list_filtered[indices_rem]
    # print 't_list_filtered  : ', t_list_filtered

    axis = 2
    indices_rem = []
    i = 0
    while i < len(t_list_filtered):
        dif_t = np.abs(avg_xyz[axis] - t_list[i][axis])
        # print 'x: ', t_list[i][axis]
        # print 'avgx: ', avg_xyz[axis]
        # print 'dif_tz: ', dif_t
        if dif_t < rej_factor:
            indices_rem.append(i)
        i += 1
    t_list_filtered = t_list_filtered[indices_rem]
    # print 't_list_filtered  : ', t_list_filtered

    # print "t-list: ", xyz_list
    # print "clean: ", t_list_filtered
    if len(t_list_filtered) < 1:
        # print 'REMOVED ALL TVEC'
        # sorted_arr = t_list[t_list[:, 0].argsort()]
        # return sorted_arr[len(sorted_arr) / 2]
        return [np.mean(xyz_list[:, 0]), np.mean(xyz_list[:, 1]), np.mean(xyz_list[:, 2])]
    else:
        # print 'REMOVED tvec: ', (len(t_list) - len(t_list_filtered))
        # print str(len(t_list_filtered)) + '/' + str(len(t_list))
        return [np.mean(t_list_filtered[:, 0]), np.mean(t_list_filtered[:, 1]), np.mean(t_list_filtered[:, 2])]


def average_orientation(q_list, axis_rej_factor, angle_rej_factor, depth = 0):
    """
    takes a list of quaternions and returns the average
    :param q_list: list of quaternions
    :param rej_factor: determines how strictly outliers are removed
    :param axis: which axis to based outlier removal on
    returns average quaternion
    if all list items are 'Outliers' and removed then return median quaternion
    """
    q_list_filtered = reject_quaternion_outliers(q_list, axis_rej_factor, angle_rej_factor)
    # if all data is removed from removing outliers we take median value

    if q_list_filtered is None:
        #TODO recurs
        print 'RECURRSION'
        if depth > 5:
            print 'BREAK!!!'
            return avg_quat_sam(q_list)
        #print 'q_list: ', q_list
        # return quaternion_median(q_list)
        return average_orientation(q_list, axis_rej_factor + 0.01, angle_rej_factor + 0.01, depth+1)
    else:
        return avg_quat_sam(q_list_filtered)


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
                move_cm = -0.07
                if current_id == self.id_back:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # adjust angle 90 degrees ccw
                    # q_rot = [0, 0.7071068, 0, 0.7071068]
                    q_rot = aa2quatSam([0, 1, 0, -np.pi / 2])
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_back_R:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # adjust angle 45 degrees ccw
                    q_rot = [0.9238795, 0, 0.3826834, 0]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_right:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # angle is good
                elif current_id == self.id_front_R:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # angle rotate 45 degrees cw
                    q_rot = [0.9238795, 0, -0.3826834, 0]  # q_rot = [0, 0.3826834, 0, 0.9238795]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_front:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # adjust angle 90 degrees cw
                    # q_rot = [0.7071068, 0, -0.7071068, 0]
                    q_rot = aa2quatSam([0, 1, 0, np.pi / 2])
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_front_L:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # angle rotate 135 degrees cw
                    q_rot = [0.3826834, 0, -0.9238795, 0]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_left:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
                    # angle rotate 180 degrees cw
                    q_rot = [0, 0, 1, 0]
                    q = mul_quaternion(q, q_rot)
                elif current_id == self.id_back_L:
                    # move IN 4cm
                    tvec = move_xyz_along_axis(tvec, q, "z", move_cm)
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

            # average orientation and position of all currently viewable markers
            t3 = time.time()
            if len(ids) > 1:
                # averaging orientation
                q_list = np.array(q_list)
                q = average_orientation(q_list, 0.1, 0.05)  # rej_factor, axis  # 1, 3
                # q = avg_quat_sam(q_list)
                # q = quatWAvgMarkley(q_list)
                # q = quaternion_median(q_list)
                # averaging position
                t_list = np.array(t_list)
                tvec = average_position(t_list, 0.01)  # rej_factor
                # (reject any positions greater than this distance away fom avg position (in meters))
            t4 = time.time()

            # average orientation and position of previous x markers
            t9 = time.time()
            if self.n_avg_previous_marker > 1:
                # average orientation
                self.marker_orient_arr.append(copy.deepcopy(q))
                self.marker_orient_arr = self.marker_orient_arr[1:self.n_avg_previous_marker + 1]
                q_list = np.array(self.marker_orient_arr)
                # TODO
                q = average_orientation(q_list, 0.05, 0.05)
                #q = avg_quat_sam(q_list)
                # average position
                self.marker_pos_arr.append(copy.deepcopy(tvec))
                self.marker_pos_arr = self.marker_pos_arr[1:self.n_avg_previous_marker + 1]
                t_list = np.array(self.marker_pos_arr)
                # tvec = average_position(t_list, 0.05)
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

        self.publish(self.outPose)

        t_total_last = time.time()
        total = "Total time: {} seconds\n"
        print(total.format(t_total_last - t_total_first))
        # # test code (ignore)
        # t = move_xyz_along_axis(tvec, q, "y", 0.5)
        # br = tf.TransformBroadcaster()
        # br.sendTransform(t, q, rospy.Time.now(), "/new", self.parent_link)

    def publish(self, pose):
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
        br.sendTransform(t, q, rospy.Time.now(), "/tracking_markers", self.parent_link)

        # move ray down to shoot from user's eye-line
        # move down x cm and forward x cm
        t = [self.eye_depth, 0, self.eye_height]
        q = [0, 0, 0, 1]
        br.sendTransform(t, q, rospy.Time.now(), "/laser_origin", "/tracking_markers")
        q_rot = [0.5, 0.5, 0.5, 0.5]
        q = mul_quaternion(q, q_rot)
        q_rot = [-0.9961947, -0.0871557, 0, 0]
        q = mul_quaternion(q, q_rot)
        br.sendTransform(t, q, rospy.Time.now(), "/usb_cam", "/tracking_markers")


def head_track():
    # TODO: select your size of marker in m
    # marker_size
    # marker_size = 0.065   # 1x1
    marker_size = 0.028  # 2x2
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
    n_previous_marker = 18  # 30 #12

    # Create object
    HT = HeadTracker(marker_size, camera_matrix, camera_distortion, parent_link, eye_height, eye_depth, image_topic,
                     n_previous_marker)

    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    head_track()
