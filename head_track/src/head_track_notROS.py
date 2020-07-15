import cv2.aruco
import numpy, cv2, sys, time, math

#define tag
id_to_find = 0
marker_size = 7

#get camera calibration
calib_path = ""
camera_matrix = numpy.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
camera_distortion = numpy.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')

#180 degree matrix rotation around x axis may not be necessary for this application
R_flip = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] = -1.0
R_flip[2,2] = -1.0

#define aruco dictionary
aruco_dict = cv2.aruco.getPredrfinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters_create()

#capture video camera, not the ros way TODO fix
cap = cv2.VideoCapture(0) #use cv bridge for ros stuff
#set camera size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:

	#read camera frame
	ret, frame = cap.read()

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
		rvec, tvec = ret[0],[0,0,:], ret[1][0,0,:]

		#draw the marker and put reference frame
		cv2.aruco.drawDetectedMarkers(frame, corners)
		cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)

	#display frame TODO rviz this
	cv2.imshow('frame', frame)

	#q to quit
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q')
		cap.release()
		cv2.destroyAllWindows()
		break
