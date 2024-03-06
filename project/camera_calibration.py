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
    def __init__(self, mtx_left, dist_left, mtx_right=None, dist_right=None):
        self.mtx_left = mtx_left
        self.dist_left = dist_left
        self.mtx_right = mtx_right if mtx_right is not None else mtx_left
        self.dist_right = dist_right if dist_right is not None else dist_left

    def split_image(self, image):
        height, width = image.shape[:2]
        mid_point = width // 2
        left_image = image[:, :mid_point]
        right_image = image[:, mid_point:]
        return left_image, right_image

    def undistort_image(self, image, camera_side="left"):
        if camera_side == "left":
            mtx = self.mtx_left
            dist = self.dist_left
        else:
            mtx = self.mtx_right
            dist = self.dist_right
        undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)
        return undistorted_img

    def process_images(
        self, left_image=None, right_image=None, concatenated_image=None
    ):
        if concatenated_image is not None:
            left_image, right_image = self.split_image(concatenated_image)
            left_undistorted = self.undistort_image(left_image, "left")
            right_undistorted = self.undistort_image(right_image, "right")
        else:
            left_undistorted = (
                self.undistort_image(left_image, "left")
                if left_image is not None
                else None
            )
            right_undistorted = (
                self.undistort_image(right_image, "right")
                if right_image is not None
                else None
            )

        if left_undistorted is not None and right_undistorted is not None:
            return np.concatenate((left_undistorted, right_undistorted), axis=1)
        elif left_undistorted is not None:
            return left_undistorted
        elif right_undistorted is not None:
            return right_undistorted
        else:
            return None
