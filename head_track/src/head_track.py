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
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.signal import savgol_filter

# cv
import cv2.aruco, cv2
from cv_bridge import CvBridge


# import pfilter
from pykalman import KalmanFilter
import pykalman
from pykalman import UnscentedKalmanFilter

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
    q0 = normalize_quat(q0) #fast_normalize_quat(q0) change
    q1 = normalize_quat(q1) #fast_normalize_quat(q1) change
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
        qr = normalize_quat(qr) #fast_normalize_quat(qr) change
        return qr

    theta_0 = np.arccos(dot)  # Since dot is in range [0, 0.9995], np.arccos() is safe
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * amount
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    qr = (s0 * q0) + (s1 * q1)
    qr = normalize_quat(qr) #fast_normalize_quat(qr) change
    return qr


def slerp_q_list(q_list):
    # slerp 'avgs' between 2 q's
    # avg between first 2, take thant, avg it aginast next, repeat
    if len(q_list) > 2:
        q_avg = slerp(q_list[0], q_list[1])
        i = 2
        while i < len(q_list):
            #print 'q_avg: ', q_avg
            #print 'q_list[i]: ', q_list[i]
            q_avg = slerp(q_avg, q_list[i])
            #print 'new_q_avg: ', q_avg
            i += 1
        return q_avg
    elif len(q_list) == 2:
        q_avg = slerp(q_list[0], q_list[1])
        return q_avg
    else:
        return q_list[0]


# BROKEN
# def median_quaternions_weiszfeld(Q, p = 1, maxAngularUpdate = 0.0001, maxIterations = 10):
#     A = np.zeros((4, 4))
#     M = Q.shape[0]
#
#     st = q_average(Q)
#     qMedian = (st[0], st[1], st[2], st[3])
#     epsAngle = 0.0000001
#     maxAngularUpdate = max(maxAngularUpdate, epsAngle)
#     theta = 10 * maxAngularUpdate
#     i = 0
#
#     while (theta > maxAngularUpdate and i <= maxIterations):
#         delta = (0, 0, 0)
#         weightSum = 0
#         j = 0
#         while j < M:
#             q = Q[j,:]
#             qj = np.array([q[0], q[1], q[2], q[3]]) * np.array(conj_quat(qMedian))
#             theta = 2 * np.arccos(qj[3])
#             if (theta > epsAngle):
#                 axisAngle = [qj[0],qj[1],qj[2]] / np.sin(theta / 2)
#                 axisAngle *= theta
#                 weight = 1.0 / pow(theta, 2 - p)
#                 delta += weight * axisAngle
#                 weightSum += weight
#             j += 1
#
#         if (weightSum > epsAngle):
#             delta /= weightSum
#             theta = np.linalg.norm(delta)
#             if (theta > epsAngle):
#                 stby2 = np.sin(theta * 0.5)
#                 delta /= theta
#                 q = (np.cos(theta * 0.5), stby2 * delta[0], stby2 * delta[1], stby2 * delta[2])
#                 qMedian = mul_quaternion(q,qMedian)
#                 #t_qfix(qMedian);
#                 qMedian = fast_normalize_quat(qMedian)
#         else:
#             theta = 0
#             i += 1
#
#     return qMedian;


def q_average(Q):
    A = np.zeros((4, 4))
    M = Q.shape[0]
    for i in range(M):
        q = Q[i,:]
        if q[3] < 0:
            q = -q
        A += np.outer(q, q)
    A /= M

    eigvals, eigvecs = np.linalg.eig(np.matmul(Q.T,Q))
    return eigvecs[:, eigvals.argmax()]

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


def quaternion_median(Q):
    # add xyzw together
    # sort that way?

    sort_arr = []
    sum = 0
    i = 0
    while i < len(Q):
        # go through all q_list
        sum += (np.abs(Q[i][0]) + np.abs(Q[i][1]) + np.abs(Q[i][2]) + np.abs(Q[i][3]))
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


def thicc(q_list, factor):
    # TODO
    #  q's that are very close to average have copies added to the list
    avg_q = slerp_q_list(q_list)
    original_q_list = q_list

    # print 'bf: ', len(q_list)
    axis = 0
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list):
        dif_q = q_list[i][axis] - avg_q[axis]
        # print 'thicc_dif_qx', dif_q
        if np.abs(dif_q) < factor:
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list[indices_rem]
    # print 'af_x: ', len(q_list_filtered)

    axis = 1
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list_filtered):
        dif_q = q_list_filtered[i][axis] - avg_q[axis]
        #print 'thicc_dif_qy', dif_q
        if np.abs(dif_q) < factor:
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list_filtered[indices_rem]
    # print 'af_y: ', len(q_list_filtered)

    axis = 2
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list_filtered):
        dif_q = q_list_filtered[i][axis] - avg_q[axis]
        #print 'thicc_dif_qz', dif_q
        if np.abs(dif_q) < factor:
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list_filtered[indices_rem]
    # print 'af_z: ', len(q_list_filtered)

    axis = 3
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list_filtered):
        dif_q = q_list_filtered[i][axis] - avg_q[axis]
        #print 'thicc_dif_qw', dif_q
        if np.abs(dif_q) < factor:
            indices_rem.append(i)
        i += 1
    q_list_filtered = q_list_filtered[indices_rem]
    print 'thiccening ', len(q_list_filtered)*3

    if len(q_list_filtered) < 1:
        # print 'REMOVED ALL (in function reject_quaternion_outliers)'
        return original_q_list
    else:
        q_list_filtered_ = np.concatenate((original_q_list,q_list_filtered))
        q_list_filtered_ = np.concatenate((q_list_filtered_, q_list_filtered))
        q_list_filtered_ = np.concatenate((q_list_filtered_, q_list_filtered))
        # print 'q_list_orig: ', original_q_list
        # print 'q_list_filt: ', q_list_filtered
        return np.array(q_list_filtered_)


def reject_quaternion_outliers(q_list, factor):
    e_list = []
    i = 0
    while i < len(q_list):
        e_list.insert(i, [euler_from_quaternion(q_list[i])[0],euler_from_quaternion(q_list[i])[1],euler_from_quaternion(q_list[i])[2]])
        e_list[i] = [e_list[i][0]+np.pi,e_list[i][1]+np.pi,e_list[i][2]+np.pi]
        i += 1
    e_list = np.array(e_list)

    avg_q = [np.mean(e_list[:, 0])-np.pi, np.mean(e_list[:, 1])-np.pi, np.mean(e_list[:, 2])-np.pi]
    # print "avg_e: ", avg_q

    axis = 0
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list):
        dif_q = avg_q[axis] - euler_from_quaternion(q_list[i])[axis]
        # temp_str = "dif_qx: {}\t< {}"
        # print temp_str.format(dif_q,factor)
        if np.abs(dif_q) < factor:
            # print "TRUE (KEEP)"
            indices_rem.append(i)
        i += 1
    # print "x says keep: ", indices_rem
    q_list_filtered1 = q_list[indices_rem]
    # print 'q-list-filtered-1: ', q_list_filtered1

    axis = 1
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list):
        dif_q = avg_q[axis] - euler_from_quaternion(q_list[i])[axis]
        # temp_str = "dif_qy: {}\t< {}"
        # print temp_str.format(dif_q, factor)
        if np.abs(dif_q) < factor:
            # print "TRUE (KEEP)"
            indices_rem.append(i)
        i += 1
    # print "y says keep: ", indices_rem
    q_list_filtered2 = q_list[indices_rem]

    axis = 2
    i = 0
    indices_rem = []  # indices we keep (NOT OUTLIERS)
    while i < len(q_list):
        dif_q = avg_q[axis] - euler_from_quaternion(q_list[i])[axis]
        # temp_str = "dif_qz: {}\t< {}"
        # print temp_str.format(dif_q, factor)
        if np.abs(dif_q) < factor:
            # print "TRUE (KEEP)"
            indices_rem.append(i)
        i += 1
    # print "z says keep: ", indices_rem
    q_list_filtered3 = q_list[indices_rem]

    q_list_final_pre = []
    q_list_final = []
    i = 0
    while i < len(q_list_filtered1):
        j = 0
        while j < len(q_list_filtered2):
            # temp_str = "{} v {}\t{} v {}\t{} v {}\t{} v {}"
            # print temp_str.format(q_list_filtered1[i][0], q_list_filtered2[j][0],q_list_filtered1[i][1],q_list_filtered2[j][1],q_list_filtered1[i][2],q_list_filtered2[j][2],q_list_filtered1[i][3],q_list_filtered2[j][3])
            if (q_list_filtered1[i][0] == q_list_filtered2[j][0]) and (q_list_filtered1[i][1] == q_list_filtered2[j][1]) and (q_list_filtered1[i][2] == q_list_filtered2[j][2]) and (q_list_filtered1[i][3] == q_list_filtered2[j][3]):
                # print "WE IN BOIS!!!"
                q_list_final_pre.append([q_list_filtered1[i][0],q_list_filtered1[i][1],q_list_filtered1[i][2], q_list_filtered1[i][3]])
            j += 1
        i += 1
    # print "!!!!!!!!!!!!!!!!!!!!!!"
    # print "q_list_final_pre: ", q_list_final_pre
    i = 0
    while i < len(q_list_final_pre):
        j = 0
        while j < len(q_list_filtered3):
            # temp_str = "{} v {}\t{} v {}\t{} v {}\t{} v {}"
            # print temp_str.format(q_list_final_pre[i][0], q_list_filtered3[j][0],q_list_final_pre[i][1], q_list_filtered3[j][1],q_list_final_pre[i][2], q_list_filtered3[j][2],q_list_final_pre[i][3],q_list_filtered3[j][3])
            if (q_list_final_pre[i][0] == q_list_filtered3[j][0]) and (q_list_final_pre[i][1] == q_list_filtered3[j][1]) and (q_list_final_pre[i][2] == q_list_filtered3[j][2]) and (q_list_final_pre[i][3] == q_list_filtered3[j][3]):
                # print "WE IN BOIS!!!"
                q_list_final.append([q_list_final_pre[i][0],q_list_final_pre[i][1],q_list_final_pre[i][2], q_list_final_pre[i][3]])
            j += 1
        i += 1
    # print "!!!!!!!!!!!!!!!!!!!!!!"
    # print "q_list_final: ", q_list_final

    # print 'remaining: ', len(q_list_final)

    # if len(q_list_filtered) < len(q_list_filtered_bf)/3:
    #     q_list_filtered = q_list_filtered_bf
    #
    # print 'after w: ', len(q_list_filtered)

    # print 'q_list: ', q_list
    # print 'q_list_filteredw: ', q_list_filtered

    if len(q_list_final) < 1:
        # print 'REMOVED ALL (in function reject_quaternion_outliers)'
        return None
    else:
        # print 'len of q_list_filtered: ', len(q_list_filtered)
        # print 'REMOVED: ', (len(q_list) - len(q_list_filtered))
        # print str(len(q_list_filtered)) + '/' + str(len(q_list))
        return np.array(q_list_final)


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
        if dif_t < rej_factor:
            indices_rem.append(i)
        i += 1
    t_list_filtered1 = t_list[indices_rem]

    axis = 1
    indices_rem = []
    i = 0
    while i < len(t_list):
        dif_t = np.abs(avg_xyz[axis] - t_list[i][axis])
        if dif_t < rej_factor:
            indices_rem.append(i)
        i += 1
    t_list_filtered2 = t_list[indices_rem]

    axis = 2
    indices_rem = []
    i = 0
    while i < len(t_list):
        dif_t = np.abs(avg_xyz[axis] - t_list[i][axis])
        if dif_t < rej_factor:
            indices_rem.append(i)
        i += 1
    t_list_filtered3 = t_list[indices_rem]

    t_list_final = []
    t_list_final_pre = []

    # IF CLOSE TO AVG ON ALL AXIS THEN WE KEEP IT
    i = 0
    while i < len(t_list_filtered1):
        j = 0
        while j < len(t_list_filtered2):
            if (t_list_filtered1[i][0] == t_list_filtered2[j][0]) and (t_list_filtered1[i][1] == t_list_filtered2[j][1]) and (t_list_filtered1[i][2] == t_list_filtered2[j][2]):
                t_list_final_pre.append([t_list_filtered1[i][0],t_list_filtered1[i][1],t_list_filtered1[i][2]])
            j += 1
        i += 1

    i = 0
    while i < len(t_list_final_pre):
        j = 0
        while j < len(t_list_filtered3):
            if (t_list_final_pre[i][0] == t_list_filtered3[j][0]) and (
                    t_list_final_pre[i][1] == t_list_filtered3[j][1]) and (
                    t_list_final_pre[i][2] == t_list_filtered3[j][2]):
                t_list_final.append([t_list_final_pre[i][0], t_list_final_pre[i][1], t_list_final_pre[i][2]])
            j += 1
        i += 1

    t_list_final = np.array(t_list_final)

    if len(t_list_final) < 1:
        return [np.mean(xyz_list[:, 0]), np.mean(xyz_list[:, 1]), np.mean(xyz_list[:, 2])]
    else:
        # print 'REMOVED tvec: ', (len(t_list) - len(t_list_filtered))
        # print str(len(t_list_filtered)) + '/' + str(len(t_list))
        return [np.mean(t_list_final[:, 0]), np.mean(t_list_final[:, 1]), np.mean(t_list_final[:, 2])]


def average_orientation(q_list, rej_factor, depth = 0):
    """
    takes a list of quaternions and returns the average
    :param q_list: list of quaternions
    :param rej_factor: determines how strictly outliers are removed
    :param depth: something f
    returns average quaternion
    if all list items are 'Outliers' and removed then return median quaternion
    """
    q_list_filtered = reject_quaternion_outliers(q_list, rej_factor)
    # if all data is removed from removing outliers we take median value

    if q_list_filtered is None:
        # print 'RECURRSION'
        if depth > 10:
            print '!!!!!!!!!!!'
            print '!!!UH OH!!!'
            print '!!!!!!!!!!!'
            #return slerp_q_list(q_list)
            return None
        # print 'q_list: ', q_list
        # return quaternion_median(q_list)
        return average_orientation(q_list, rej_factor + 0.15, depth + 1)
    else:
        return slerp_q_list(q_list_filtered)


def test_function_Savitzky_Golay(q_list):
    # e_list = []
    # i = 0
    # while i < len(q_list):
    #     e_list.insert(i, [euler_from_quaternion(q_list[i])[0], euler_from_quaternion(q_list[i])[1],
    #                       euler_from_quaternion(q_list[i])[2]])
    #     e_list[i] = [e_list[i][0] + np.pi, e_list[i][1] + np.pi, e_list[i][2] + np.pi]
    #     i += 1
    # e_list = np.array(e_list)
    #
    # avg_q = [np.mean(e_list[:, 0]) - np.pi, np.mean(e_list[:, 1]) - np.pi, np.mean(e_list[:, 2]) - np.pi]
    # scipy.savgol_filter()
    q_list = np.array(q_list)
    x_axis = np.arange(1, len(q_list)+1, 1)  # x axis

    window_len = len(x_axis)
    if window_len % 2 == 0:
        window_len -= 1
    yx = savgol_filter(q_list[:,0], window_len, 2)
    yy = savgol_filter(q_list[:,1], window_len, 2)
    yz = savgol_filter(q_list[:,2], window_len, 2)
    yw = savgol_filter(q_list[:,3], window_len, 2)

    return_list = []
    i = 0
    while i < len(x_axis):
        return_list.append([yx[i],yy[i],yz[i],yw[i]])
        i += 1
    return_list = np.array(return_list)

    # plt.plot(x_axis, q_list[:,0], linewidth=2, linestyle="-", c="r")
    # plt.plot(x_axis, q_list[:,1], linewidth=2, linestyle="-", c="g")
    # plt.plot(x_axis, q_list[:,2], linewidth=2, linestyle="-", c="b")
    # plt.plot(x_axis, q_list[:,3], linewidth=2, linestyle="-", c="y")

    plt.plot(x_axis, yx, linewidth=2, linestyle=":", c="r")
    plt.plot(x_axis,  yy, linewidth=2, linestyle=":", c="g")
    plt.plot(x_axis,  yz, linewidth=2, linestyle=":", c="b")
    plt.plot(x_axis,  yw, linewidth=2, linestyle=":", c="y")

    #avg_q = slerp_q_list(return_list)
    #avg_q = quatWAvgMarkley(return_list)
    # weights = []
    # i = 0.0
    # while i < len(x_axis):
    #     if i < len(x_axis)/2:
    #         weights.append(i/len(x_axis))
    #         print i/len(x_axis)
    #     else:
    #         weights.append((len(x_axis) - i)/len(x_axis))
    #         print (len(x_axis) - i)/len(x_axis)
    #     i += 1.0
    # print 'weights: ', weights
    avg_q = q_average(return_list)
    #avg_q = quatWAvgMarkley(return_list,weights)
    #avg_q = quaternion_median(return_list)

    list = []
    i = 0
    while i < len(x_axis):
        list.append(avg_q)
        i += 1
    list = np.array(list)

    plt.plot(x_axis, list[:, 0], linewidth=2, linestyle="--", c="r")
    plt.plot(x_axis, list[:, 1], linewidth=2, linestyle="--", c="g")
    plt.plot(x_axis, list[:, 2], linewidth=2, linestyle="--", c="b")
    plt.plot(x_axis, list[:, 3], linewidth=2, linestyle="--", c="y")

    return avg_q


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
        self.f = open("quat_data.txt", "w")

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
                tvec = [tvec[0] / 10, tvec[1] / 10, tvec[2] / 10]
                move_cm = -0.175 / 4    #idk why this has to be divided by 4 instead of by 2 like id expect
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

            # # TODO
            # # HARDCODE TEST ORIENTATION AND POSITIONS
            # q = [0, 0, 0, 1]
            # q_list = [[0, 0, 0, 1],[0, 0, 0.0871557, 0.9961947],[0, 0, -0.0871557, 0.9961947],[ 0, 0, -0.3826834, 0.9238795],[0, 0, 0.3826834, 0.9238795],[0, 0, 0.5735764, 0.819152],[0, 0, -0.5735764, 0.819152]]
            # q_list = [[0,0,0,-1],[1,0,0,0],[0, 0, 0, 1]]

            # self.f.write("Raw Quaternions from Hat:\n-------------------\n")
            # q_list = np.array(q_list)
            # i = 0
            # while i < len(q_list):
            #     self.f.write("Q: ")
            #     self.f.write(str(q_list[i]))
            #     self.f.write("\tE: ")
            #     self.f.write(str(euler_from_quaternion(q_list[i])))
            #     self.f.write("\n")
            #     i += 1
            # self.f.write("\n\n")

            # average orientation and position of all currently viewable markers
            t3 = time.time()
            if len(ids) > 1:
                # averaging orientation
                q_list = np.array(q_list)

                # # TEMP TODO
                # e_list = []
                # i = 0
                # while i < len(q_list):
                #     e_list.insert(i, [euler_from_quaternion(q_list[i])[0], euler_from_quaternion(q_list[i])[1],
                #                       euler_from_quaternion(q_list[i])[2]])
                #     e_list[i] = [e_list[i][0] + np.pi, e_list[i][1] + np.pi, e_list[i][2] + np.pi]
                #     i += 1
                # e_list = np.array(e_list)
                # avg_q = [np.mean(e_list[:, 0]) - np.pi, np.mean(e_list[:, 1]) - np.pi, np.mean(e_list[:, 2]) - np.pi]
                # self.f.write("Avg E: \t\t")
                # self.f.write(str(avg_q))
                # self.f.write("\n")

                # q_list = reject_quaternion_outliers(q_list, 1)    # reject a bit garbage
                #
                # self.f.write("Filtered Quaternions:\n-------------------\n")
                # q_list = np.array(q_list)
                # i = 0
                # while i < len(q_list):
                #     self.f.write("Q: ")
                #     self.f.write(str(q_list[i]))
                #     self.f.write("\tE: ")
                #     self.f.write(str(euler_from_quaternion(q_list[i])))
                #     self.f.write("\n")
                #     i += 1
                # self.f.write("\n\n")
                #
                # # q_list = thicc(q_list,0.01)     # beef up decent numbers
                # q = slerp_q_list(q_list)   # slerp them numbers

                # test_function_Savitzky_Golay(q_list)
                # plt.show()
                # plt.close()

                # # Averaging Orientation
                # q = average_orientation(q_list, 3*np.pi/180)
                # if q is None:
                #     # print 'prev_q: ', self.marker_orient_arr
                #     # print 'q we grab: ', self.marker_orient_arr[self.n_avg_previous_marker - 1]
                #     #q = self.marker_orient_arr[self.n_avg_previous_marker - 1]
                #     temp = np.array(copy.deepcopy(self.marker_orient_arr))
                #     q = slerp_q_list(temp)
                if len(q_list) > 2:
                    q = test_function_Savitzky_Golay(q_list)
                else:
                    q = q_average(q_list)
                # plt.show()
                # plt.close()

                # q = slerp_q_list(smoothed_state_means)

                # averaging position
                t_list = np.array(t_list)
                tvec = average_position(t_list, 0.03)  # rej_factor
                # (reject any positions greater than this distance away fom avg position (in meters))
            t4 = time.time()

            # self.f.write("Average Quaternion from Filtered Quaternions (SLERP):\n-------------------\n")
            # self.f.write("Q: ")
            # self.f.write(str(q))
            # self.f.write("\tE: ")
            # self.f.write(str(euler_from_quaternion(q)))
            # self.f.write("\n\n")

            # average orientation and position of previous x markers
            t9 = time.time()
            if self.n_avg_previous_marker > 1:
                # average orientation
                self.marker_orient_arr.append(copy.deepcopy(q))
                self.marker_orient_arr = self.marker_orient_arr[1:self.n_avg_previous_marker + 1]
                q_list = np.array(self.marker_orient_arr)
                # TODO
                #q = average_orientation(q_list, 0.1)
                #q_list = reject_quaternion_outliers(q_list,0.8)
                q = quatWAvgMarkley(q_list) # <---what i was using

                # q_list = reject_quaternion_outliers(q_list, 0.8)    # reject some more garbage
                # q = slerp_q_list(q_list)    # slerp 'em
                # q = slerp(q, quatWAvgMarkley(q_list),0.1)

                # q = slerp_q_list(q_list)

                #q = avg_quat_sam(q_list)
                # average position
                self.marker_pos_arr.append(copy.deepcopy(tvec))
                self.marker_pos_arr = self.marker_pos_arr[1:self.n_avg_previous_marker + 1]
                t_list = np.array(self.marker_pos_arr)
                # tvec = average_position(t_list, 0.05)
                tvec = [np.average(t_list[:, 0]), np.average(t_list[:, 1]), np.average(t_list[:, 2])]
            t10 = time.time()

            # temp_str = "Quaternions from Previous {} Poses:\n-------------------\n"
            # self.f.write(temp_str.format(self.n_avg_previous_marker))
            # i = 0
            # while i < len(q_list):
            #     self.f.write("Q: ")
            #     self.f.write(str(q_list[i]))
            #     self.f.write("\tE: ")
            #     self.f.write(str(euler_from_quaternion(q_list[i])))
            #     self.f.write("\n")
            #     i += 1
            # self.f.write("\n\n")
            #
            # self.f.write("Published Quaternion slerped from Previous Poses:\n-------------------\n")
            # self.f.write("Q: ")
            # self.f.write(str(q))
            # self.f.write("\tE: ")
            # self.f.write(str(euler_from_quaternion(q)))
            # self.f.write("\n\n")
            #
            # self.f.write("ROSTIME: ")
            # self.f.write(str(rospy.Time.now()))
            #
            # self.f.write("\n\n\n")

            #print 'q_list: ', q_list
            #print 'q: ', q

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
        # print 'pose: ', self.outPose

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
        # q = mul_quaternion(q, q_rot)
        # q_rot = [-0.9961947, -0.0871557, 0, 0]
        q = mul_quaternion(q, q_rot)
        br.sendTransform(t, q, rospy.Time.now(), "/usb_cam", "/tracking_markers")


def head_track():
    # TODO: select your size of marker in m
    #  (oh yeah and multiply it by 10, idk why but it tracks better, i divide distance by 10 later)
    # marker_size
    # marker_size = 0.065   # 1x1
    marker_size = 0.03 * 10  # 2x2
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
    n_previous_marker = 15  # 30 #12

    # Create object
    HT = HeadTracker(marker_size, camera_matrix, camera_distortion, parent_link, eye_height, eye_depth, image_topic,
                     n_previous_marker)

    rate = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    head_track()
