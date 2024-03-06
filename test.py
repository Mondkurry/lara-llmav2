import cv2
import numpy as np
import glob
import json

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def stack_images_vertically(image_path1, image_path2):
    # Load the images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Determine the target width to match the smallest width of the two images
    target_width = min(img1.width, img2.width)

    # Resize images to have the same width while maintaining aspect ratios
    img1 = img1.resize(
        (target_width, int(target_width * img1.height / img1.width)),
        Image.Resampling.LANCZOS,
    )
    img2 = img2.resize(
        (target_width, int(target_width * img2.height / img2.width)),
        Image.Resampling.LANCZOS,
    )

    # Create a new image with height equal to the sum of both images' heights
    combined_height = img1.height + img2.height
    combined_image = Image.new("RGB", (target_width, combined_height))

    # Paste the images into the combined image
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (0, img1.height))

    # Display the combined image
    plt.figure(figsize=(5, 10))
    plt.imshow(combined_image)
    plt.axis("off")  # Hide axes
    plt.show()


# Example usage:
# stack_images_vertically('path_to_image_1.jpg', 'path_to_image_2.jpg')


class CameraCalibrator:
    def __init__(self, checkerboard, image_dir, criteria):
        self.checkerboard = checkerboard
        self.image_dir = image_dir
        self.criteria = criteria
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        self._prepare_object_points()

    def _prepare_object_points(self):
        objp = np.zeros((1, self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objp[0, :, :2] = np.mgrid[
            0 : self.checkerboard[0], 0 : self.checkerboard[1]
        ].T.reshape(-1, 2)
        self.objp = objp

    def find_corners(self):
        images = glob.glob(self.image_dir)
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
            if ret == True:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )
                self.imgpoints.append(corners2)
                # Save the image size
                self.img_shape = gray.shape[::-1]
                cv2.drawChessboardCorners(img, self.checkerboard, corners2, ret)
                cv2.imshow("img", img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

    def calibrate_camera(self):
        if not hasattr(self, "img_shape"):
            raise ValueError("No images processed. Please run 'find_corners' first.")

        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            self.objpoints,
            self.imgpoints,
            self.img_shape,
            np.zeros((3, 3)),
            np.zeros((4, 1)),
            None,
            None,
            flags=(
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                + cv2.fisheye.CALIB_CHECK_COND
                + cv2.fisheye.CALIB_FIX_SKEW
            ),
            criteria=self.criteria,
        )
        if ret:
            print(f"Found {len(self.objpoints)} valid images for calibration")
            print(f"DIM={self.img_shape}")
            print(f"K={mtx.tolist()}")
            print(f"D={dist.tolist()}")
            self.save_calibration(mtx, dist)
        else:
            print("Calibration failed.")

    def save_calibration(self, mtx, dist):
        calibration_data = {"K": mtx.tolist(), "D": dist.tolist()}
        with open("calibration_data.json", "w") as f:
            json.dump(calibration_data, f)


class ImageUndistorter:
    def __init__(self, calibration_file, target_size=(1920, 1080)):
        self.calibration_data = self.load_calibration(calibration_file)
        self.target_size = target_size  # Manually set the target size for input images

    def load_calibration(self, calibration_file):
        with open(calibration_file, "r") as f:
            return json.load(f)

    def resize_image(self, img):
        """
        Resize the input image to the target size.
        """
        return cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)

    def undistort_image(self, image_path):
        # Read the distorted image
        img = cv2.imread(image_path)
        # Resize it to the target size
        img = self.resize_image(img)
        # Get the size of the resized image
        h, w = img.shape[:2]

        # Get the camera matrix and distortion coefficients from the calibration data
        mtx = np.array(self.calibration_data["K"])
        dist = np.array(self.calibration_data["D"])

        # Calculate the new optimal camera matrix for undistortion
        new_K = mtx.copy()
        # Adjust the new camera matrix to be based on the width and height of the resized image
        new_K[0, 0] *= w / mtx[0, 2] / 2
        new_K[1, 1] *= h / mtx[1, 2] / 2
        new_K[0, 2] = w / 2.0
        new_K[1, 2] = h / 2.0

        # Initialize the undistortion transformation map
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            mtx, dist, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )

        # Apply the undistortion transformation
        undistorted_img = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return undistorted_img

    def display_undistorted_image(self, undistorted_img):
        # Display the undistorted image
        cv2.imshow("Undistorted Image", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class ImageConcatenator:
    def __init__(self):
        pass

    def concatenate_images(self, img_left, img_right):
        """
        Concatenates two images side by side.
        """
        # Ensure the images have the same height before concatenation
        h_left = img_left.shape[0]
        h_right = img_right.shape[0]
        if h_left != h_right:
            # Resize images to the smallest height
            h_new = min(h_left, h_right)
            img_left = cv2.resize(
                img_left, (int(img_left.shape[1] * h_new / h_left), h_new)
            )
            img_right = cv2.resize(
                img_right, (int(img_right.shape[1] * h_new / h_right), h_new)
            )

        # Concatenate images horizontally
        concatenated_image = np.hstack((img_left, img_right))
        return concatenated_image

    def display_concatenated_image(self, concatenated_image):
        """
        Displays the concatenated image.
        """
        cv2.imshow("Concatenated Image", concatenated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_concatenated_image(self, concatenated_image, output_path):
        """
        Saves the concatenated image to the specified output path.
        """
        cv2.imwrite(output_path, concatenated_image)


def run(calibrate=True, undistort=True, output=False):
    if calibrate:
        # Usage
        calibrator = CameraCalibrator(
            (6, 10),
            "images/Left_Images_1080/*.png",
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        calibrator.find_corners()
        calibrator.calibrate_camera()
    if undistort:
        undistorter = ImageUndistorter(
            "calibration_data.json", target_size=(1920, 1080)
        )
        # undistorter.undistort_image("images/Left_Images_240/imgLeft_0.png")
        # undistorter.undistort_image("images/Left_Images_1080/imgLeft_0.png")
        img_1 = undistorter.undistort_image("image_from_dataset_right.png")
        img_2 = undistorter.undistort_image("image_from_dataset_left.png")

        # Concatenate the undistorted images
        concatenator = ImageConcatenator()
        concatenated_image = concatenator.concatenate_images(img_2, img_1)
        concatenator.display_concatenated_image(concatenated_image)
        concatenator.save_concatenated_image(
            concatenated_image, "concatenated_image.png"
        )
    if output:
        stack_images_vertically("image_from_dataset.png", "concatenated_image.png")


def main():
    # run(calibrate=True, undistort=True)
    # run(calibrate=False, undistort=True, output=False)
    run(calibrate=False, undistort=False, output=True)


if __name__ == "__main__":
    main()
