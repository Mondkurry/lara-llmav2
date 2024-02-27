from camera_calibration import show_camera, CameraCalibrator, ImageProcessor
import cv2


def run_calibration():
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


def main():
    try:
        show_camera()
    except:
        print("Error in show_camera()")


if __name__ == "__main__":
    main()
