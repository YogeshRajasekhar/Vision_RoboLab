import cv2
import numpy as np


class ArucoDetect:
    def __init__(self,camera_matrix,dist_coeff,marker_length=0.1):
        self.camera_matrix = camera_matrix
        self.dist_coeff = dist_coeff
        self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.charuco_params=cv2.aruco.CharucoParameters()
        self.refined_params=cv2.aruco.RefineParameters()
        self.aruco_marker_length = marker_length
        self.marker_location = dict()
        self.squares_x=5#
        self.squares_y=7#
        self.square_length=0.04
        self.marker_length=0.02
        self.board = cv2.aruco.CharucoBoard((self.squares_y,self.squares_x),self.square_length,self.marker_length,self.aruco_dict,None)
        self.charuco_detector =  cv2.aruco.CharucoDetector(self.board,self.charuco_params,self.parameters,self.refined_params)

    def get_depth_from_map(x,y):
        return (x+y)//2
    
    def detect_marker(self,frame):
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        if len(corners) > 0:
            pass
        for i,(mark_corner,mark_id) in enumerate(zip(corners,ids)):
            corner=mark_corner.reshape(4,2)
            (top_left, top_right, bottom_right, bottom_left) = corner
            
            top_right = np.array((int(top_right[0]), int(top_right[1])))
            bottom_right = np.array((int(bottom_right[0]), int(bottom_right[1])))
            bottom_left = np.array((int(bottom_left[0]), int(bottom_left[1])))
            top_left = np.array((int(top_left[0]), int(top_left[1])))

            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            center_x=int(np.array([top_left[0]+bottom_right[0]])/2)
            center_y=int(np.array([top_left[1]+bottom_right[1]])/2)

            right_line=np.int8((top_right+bottom_right)/2)
            top_line=np.int8((top_left+top_right)/2)
            pt_right=np.int8((0.7*np.array([center_x,center_y]))+(0.3*right_line))
            pt_up=np.int8((0.7*np.array([center_x,center_y]))+(0.3*top_line))

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.aruco_marker_length, camera_matrix, dist_coeff
                )
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeff, rvec[i], tvec[i], 0.03)
        return frame
    
    def detect_charucoMarker(self,frame):
        charuco_corner,charuco_id, marker_corners,  marker_ID = self.charuco_detector.detectBoard(frame)
        if charuco_corner is not None and charuco_id is not None and len(charuco_id) > 3:
            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corner,
                charuco_id,
                self.board,
                self.camera_matrix,
                self.dist_coeff
            )

            if retval:
                # Draw the pose (axes) at the board center
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeff, rvec, tvec, 0.05)  # 5cm axis

                # Optionally draw ChArUco corners
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corner, charuco_id)
        return frame

if __name__=='__main__':
    camera_matrix = np.array([[2,0,1],
                            [0,2,1],
                            [0,0,1]])
    dist_coeff=np.array([0.5,0.4,0.5,0.5,0.8])

    aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    marker_length = 0.1

    frame=cv2.imread(r'marker.png')

    