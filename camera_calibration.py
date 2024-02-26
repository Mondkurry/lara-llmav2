import numpy as np
import cv2 as cv2


def show_camera():
    """Use openCV to show the current camera frame in a continuous loop. Press 'q' to exit the loop."""
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()  # ret, frame = cap.read()

        # cv.putText(Frame, Text, Org, FontFace, FontScale, Color, Thickness, LineType, BottomLeftOrigin)
        cv2.putText(
            frame,
            "Press q to exit",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


class CameraCalibrator:

    """
    Class for calibrating a camera using a chessboard pattern.

    Attributes:
        checkerboard_size (tuple): The number of inner corners per a chessboard row and column.
        square_size (float): The size of one square on the chessboard (usually in meters or inches).
        objpoints (list): The list that stores the 3D points of the checkerboard corners in the real world.
        imgpoints (list): The list that stores the 2D points of the checkerboard corners in the image plane.

    Methods:
        find_checkerboard(image):
            Finds and stores the chessboard corners in the provided image.

            Args:
                image (numpy.ndarray): The image in which to find the chessboard corners.

            Returns:
                tuple: A boolean indicating success and the corners found as a numpy array.

        calibrate_camera(image_size):
            Calibrates the camera using the stored 3D-2D point correspondences.

            Args:
                image_size (tuple): The size of the calibration images (width, height).

            Returns:
                tuple: The camera matrix and distortion coefficients.
    """

    def __init__(self, checkerboard_size, square_size):
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.objp = np.zeros(
            (1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32
        )
        self.objp[0, :, :2] = np.mgrid[
            0 : checkerboard_size[0], 0 : checkerboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size

    def find_checkerboard(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        if ret:
            self.imgpoints.append(corners)
            self.objpoints.append(self.objp)
        return ret, corners

    def calibrate_camera(self, image_size):
        _, mtx, dist, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )
        return mtx, dist


class ImageProcessor:
    """
    Class for processing images using camera calibration data.

    Attributes:
        mtx (ndarray): Camera matrix from calibration.
        dist (ndarray): Distortion coefficients from calibration.

    Methods:
        split_image(image):
            Splits the image into left and right halves.

            Parameters:
                image (ndarray): Image to be split.

            Returns:
                tuple: Left and right halves of the image.

        undistort_image(image):
            Removes distortion from an image using calibration data.

            Parameters:
                image (ndarray): Image to be undistorted.

            Returns:
                ndarray: The undistorted image.

        concatenate_images(left_image, right_image):
            Concatenates two images horizontally.

            Parameters:
                left_image (ndarray): Left half of the image.
                right_image (ndarray): Right half of the image.

            Returns:
                ndarray: The concatenated image.
    """

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

    def split_image(self, image):
        _, w = image.shape[:2]
        return image[:, : w // 2], image[:, w // 2 :]

    def undistort_image(self, image):
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h)
        )
        undistorted_img = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
        x, y, w, h = roi
        undistorted_img = undistorted_img[y : y + h, x : x + w]
        return undistorted_img

    def concatenate_images(self, left_image, right_image):
        return np.concatenate((left_image, right_image), axis=1)


def main():
    show_camera()

    # Assume we have calibration images for left and right cameras
    calibration_images_left = ["ADD PATH HERE"]  # list of file paths or images
    calibration_images_right = ["ADD PATH HERE"]  # list of file paths or images

    # Calibrate the left camera
    calibrator_left = CameraCalibrator(
        (9, 6), 0.025
    )  # Checkerboard of 9x6 squares, each 0.025m
    for img_path in calibration_images_left:
        img = cv2.imread(img_path)
        calibrator_left.find_checkerboard(img)
    mtx_left, dist_left = calibrator_left.calibrate_camera(img.shape[1::-1])

    # Calibrate the right camera
    calibrator_right = CameraCalibrator((9, 6), 0.025)
    for img_path in calibration_images_right:
        img = cv2.imread(img_path)
        calibrator_right.find_checkerboard(img)
    mtx_right, dist_right = calibrator_right.calibrate_camera(img.shape[1::-1])

    # Process an example image
    image_processor_left = ImageProcessor(mtx_left, dist_left)
    image_processor_right = ImageProcessor(mtx_right, dist_right)

    # Read the image
    image = cv2.imread("/mnt/data/image.png")

    # Split the image
    left_image, right_image = image_processor_left.split_image(image)

    # Undistort the images
    # Undistort the images
    left_undistorted = image_processor_left.undistort_image(left_image)
    right_undistorted = image_processor_right.undistort_image(right_image)

    # Concatenate the undistorted images
    undistorted_image = image_processor_left.concatenate_images(
        left_undistorted, right_undistorted
    )

    # Display the result
    cv2.imshow("Undistorted Image", undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the undistorted image if needed
    cv2.imwrite("undistorted_image.png", undistorted_image)


if __name__ == "__main__":
    main()
