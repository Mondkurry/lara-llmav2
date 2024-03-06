import cv2
import os
from project.camera_calibration import show_camera, CameraCalibrator, ImageProcessor


def run_calibration(run_calibration: bool = False, run_processor: bool = False):
    if run_calibration:
        # Assume we have calibration images for left and right cameras
        calibration_images_left = [
            "images/left_images_1080/imgLeft_0.png",
            "images/left_images_1080/imgLeft_1.png",
            "images/left_images_1080/imgLeft_2.png",
            "images/left_images_1080/imgLeft_3.png",
            "images/left_images_1080/imgLeft_4.png",
            "images/left_images_1080/imgLeft_5.png",
            "images/left_images_1080/imgLeft_6.png",
            "images/left_images_1080/imgLeft_7.png",
            "images/left_images_1080/imgLeft_8.png",
            "images/left_images_1080/imgLeft_9.png",
        ]

        calibration_images_right = [
            "images/right_images_1080/imgRight_0.png",
            "images/right_images_1080/imgRight_1.png",
            "images/right_images_1080/imgRight_2.png",
            "images/right_images_1080/imgRight_3.png",
            "images/right_images_1080/imgRight_4.png",
            "images/right_images_1080/imgRight_5.png",
            "images/right_images_1080/imgRight_6.png",
            "images/right_images_1080/imgRight_7.png",
            "images/right_images_1080/imgRight_8.png",
            "images/right_images_1080/imgRight_9.png",
        ]

        # Calibrate the left camera
        calibrator_left = CameraCalibrator((10, 6), 0.035)
        for img_path in calibration_images_left:
            if not os.path.exists(img_path):
                print(f"Image path does not exist: {img_path}")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image at {img_path}")
                continue

            ret, corners = calibrator_left.find_checkerboard(img)
            if not ret:
                print(f"Checkerboard not found in {img_path}")
                continue

        mtx_left, dist_left = calibrator_left.calibrate_camera(img.shape[1::-1])

        # Calibrate the right camera
        calibrator_right = CameraCalibrator((9, 6), 0.025)
        for img_path in calibration_images_right:
            img = cv2.imread(img_path)
            calibrator_right.find_checkerboard(img)
        mtx_right, dist_right = calibrator_right.calibrate_camera(img.shape[1::-1])

        print("Left camera matrix:")
        print(mtx_left)
        print("Left distortion coefficients:")
        print(dist_left)

        print("Right camera matrix:")
        print(mtx_right)
        print("Right distortion coefficients:")
        print(dist_right)

    if run_processor:
        try:
            image_processor = ImageProcessor(mtx_left, dist_left, mtx_right, dist_right)
        except NameError:
            print("Please run the calibration first")
            return

        # Example for processing separate images
        left_image = cv2.imread("images/left images/imgLeft_0.png")
        right_image = cv2.imread("images/right images/imgRight_0.png")

        if left_image or right_image is None:
            print("Image did not populate")

        undistorted_image = image_processor.process_images(
            left_image=left_image, right_image=right_image
        )

        # Example for processing a concatenated image
        # concatenated_image = cv2.imread("path/to/concatenated_image.png")
        # undistorted_image = image_processor.process_images(concatenated_image=concatenated_image)

        # Display the result
        cv2.imshow("Undistorted Image", undistorted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the undistorted image if needed
        cv2.imwrite("undistorted_image.png", undistorted_image)


def main():
    try:
        run_calibration(run_calibration=True, run_processor=True)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
