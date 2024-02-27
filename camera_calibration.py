import numpy as np
import cv2 as cv2


def show_camera():
    """Use openCV to show the current camera frame in a continuous loop. Press 'esc' to exit the loop and Spacebar to take a picture."""

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Calibration Image Capture")
    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "images/calibration_image_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
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
