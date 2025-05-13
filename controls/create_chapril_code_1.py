import cv2
import numpy as np
import apriltag
import cairosvg
import os
def create_custom_chessboard_with_images(rows, cols, square_size, images, output_filename):
    """
    Creates a chessboard-like grid with custom images inserted into white squares only.

    :param rows: Number of rows in the grid.
    :param cols: Number of columns in the grid.
    :param square_size: Pixel size of each square.
    :param image_files: List of image file paths to use.
    :param output_filename: Output image file name.
    """
    board_image = np.ones((cols * square_size, rows * square_size, 3), dtype=np.uint8) * 255
    print((cols,rows),board_image.shape)
    count=0
    for i in range(cols):
        for j in range(rows):
            if (i+j)%2==0:
                board_image[i*square_size:(i+1)*square_size,j*square_size:(j+1)*square_size]=np.array([0,0,0])
            else:
                code=images[count]
                code=cv2.resize(code,(square_size//2,square_size//2))
                count+=1
                board_image[int((i+0.25)*square_size):int((i+0.75)*square_size),int((j+0.25)*square_size):int((j+0.75)*square_size)]=code
    board_image = cv2.copyMakeBorder(
    board_image,
    top=50, bottom=50,    # pad 50 pixels on top and bottom
    left=50, right=50,    # pad 30 pixels on left and right
    borderType=cv2.BORDER_CONSTANT,
    value=[255, 255, 255])  # white padding (BGR)``)

    # Save the final board image
    cv2.imwrite(output_filename, board_image)
    print(f"Board saved to {output_filename}")


    # Save the generated image
    cv2.imwrite(output_filename, board_image)

    # Display the generated image
    cv2.imshow('Custom Chessboard with Images', board_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_april_tags(num_tags,order=None):
    lst=[]
    if order is not None:
        lts=order
    else:
        lts=range(num_tags)
    for i in lts:
        png_bytes = cairosvg.svg2png(url=f'./april_tags/tag36h11-{i}.svg')
    
    # Convert PNG bytes to NumPy array
        image_array = np.frombuffer(png_bytes, dtype=np.uint8)
        
        # Decode image as OpenCV image
        img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)[38:-38,38:-38,:3]
        lst.append(img)
    return lst
# List of image file paths (replace with your own images)

# Define the board dimensions (3x4 grid)

rows = 4
cols = 6
square_size = 150  # Size of each square in pixels
images = generate_april_tags(rows*cols)
# Output file path for the generated image
output_filename = 'custom_chessboard_with_images.png'

# Create the custom chessboard with images
create_custom_chessboard_with_images(rows, cols, square_size, images, output_filename)
