# You should replace these 3 lines with the output in calibration step
import numpy as np
import cv2
import sys
import json

image_path = "image_from_dataset_right.png"
calibration_file = "calibration_data.json"


def load_calibration(calibration_file):
    with open(calibration_file, "r") as f:
        return json.load(f)


calibration_data = load_calibration(calibration_file)

# Measure from image
DIM = cv2.imread(image_path).shape[:2][::-1]  # (width, height)
K = np.array(calibration_data["K"])
D = np.array(calibration_data["D"])


def undistort(img_path, balance=0.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, D, dim2, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        undistort(p)
