import cv2
import cv2.aruco as aruco

# Define board parameters
squares_x = 5        # number of chessboard squares along X
squares_y = 7        # number of chessboard squares along Y
square_length = 0.04 # in meters or any unit (e.g., 4cm)
marker_length = 0.02 # marker side length (smaller than square length)

# Choose dictionary (DICT_4X4_50 is common)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
print(aruco_dict)

board= aruco.CharucoBoard((squares_x,squares_y),square_length,marker_length,aruco_dict)

# Image size in pixels (for printing, higher is better)
img_size = (1000, 1400)  # width x height in pixels

# Draw board to image
board_img = board.generateImage(img_size)

# Save the image
cv2.imwrite("charuco_board.png", board_img)

print("ChArUco board saved as 'charuco_board.png'")
