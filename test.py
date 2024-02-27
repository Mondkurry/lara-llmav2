import numpy as np


def main():
    square_size = 1.0  # Example: 1 inch or 1 centimeter
    checkerboard_size = (9, 6)

    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    print(objp, objp.shape)
    objp[:, :2] = (
        np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(-1, 2)
        * square_size
    )
    print("after transform")
    print(objp, objp.shape)


if __name__ == "__main__":
    main()
