#!/usr/bin/env python


#TEST VERSION NOT FINISHED CODE


import rospy, cv2.aruco, numpy, cv2, sys, time, math, tf, sensor_msgs.msg, geometry_msgs.msg
from cv_bridge import CvBridge


#define tag
id_to_find = 1
# marker_size = 7
marker_size = 10 #m -- we dividde by 100 later for accuracy and speed


#initalize these values here (will be in init)
tvec = [0,0,0]
q = [0,0,0,1]

#get camera calibration


calib_path = '/home/csrobot/catkin_ws/src/head_track/calibration/'
camera_matrix = numpy.loadtxt(calib_path + 'cameraMatrix.txt', delimiter=',')
camera_distortion = numpy.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')

#180 degree matrix rotation around x axis   may not be necessary for this application
R_flip = numpy.zeros((3,3), dtype=numpy.float32)
R_flip[0,0] = 1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0

#define aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create()

#set up link between ros topics and opencv
bridge = CvBridge()

#TODO set up image publisher
pubIm = rospy.Publisher("/detected_frame", sensor_msgs.msg.Image, queue_size=10)
pubPose = rospy.Publisher("/head_pose", geometry_msgs.msg.Pose, queue_size=10)

pose = geometry_msgs.msg.Pose()


def callback(rawFrame): #gets frame from realsense
	# global temporarily -- make a class later
	global tvec
	global q
	
	#capture video camera frame
	#cap = bridge.imgmsg_to_cv2(rawFrame, "bgr8") #desired_encoding='passthrough')
	frame = bridge.imgmsg_to_cv2(rawFrame, "bgr8")
	#set camera size
	#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	#read camera frame
	#ret, frame = cap.read()

	#grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#find aruco markers in that mf image
	corners, ids, rejected = cv2.aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)

	if ids != None and ids[0] == id_to_find:

		#ret = [rvec, tvec, ?]
		#array of rotation and position of each marker
		#rvec = [[rvec1],[rvec2],...] attitude
		#tvec = [[tvec1].[tvec2],...] position
		ret = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

		#unpack output only get first
		rvec, tvec = ret[0][0,0,:], ret[1][0,0,:] #looks like this can be eliminated if we put the two vars into the above function call, doesnt hurt us to see more markers?

		#draw the marker and put reference frame
		cv2.aruco.drawDetectedMarkers(frame, corners)
		cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

		#assemble pose message
		#rospy.logwarn("rvec: %f %f %f\ntvec %d %d %d\n", rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2])
		q = tf.transformations.quaternion_from_euler(rvec[0], -rvec[2], rvec[1]) # the rotation was wierd so we sawp y and z and make y negative

		
		pose.position.x = tvec[0]
		pose.position.y = tvec[1]
		pose.position.z = tvec[2]
		pose.orientation.x = q[0]
		pose.orientation.y = q[1]
		pose.orientation.z = q[2]
		pose.orientation.w = q[3]

		tvec[0] = tvec[0]/100
		tvec[1] = tvec[1]/100
		tvec[2] = tvec[2]/100

	#Added by sam
	#publish tf frame that matches pose
	br = tf.TransformBroadcaster()
	br.sendTransform(tvec, q, rospy.Time.now(), "/laser_origin", "/camera_link")

	#not being used atm
	#display frame TODO rviz this
	pubIm.publish(bridge.cv2_to_imgmsg(frame, encoding="passthrough"))
	pubPose.publish(pose)
	#cv2.imshow("Image window", frame)

	


def head_track():
	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber("/camera/color/image_raw", sensor_msgs.msg.Image, callback)
	rospy.spin()

if __name__ == '__main__':
	head_track()
