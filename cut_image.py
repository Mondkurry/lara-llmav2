import cv2
import os


def split_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: The image at {image_path} could not be read.")
        return

    _, w = img.shape[:2]

    midpoint = w // 2

    left_img = img[:, :midpoint]
    right_img = img[:, midpoint:]

    base_path, ext = os.path.splitext(image_path)
    left_path = f"{base_path}_left.png"
    right_path = f"{base_path}_right.png"

    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)

    print(f"Left image saved as: {left_path}")
    print(f"Right image saved as: {right_path}")


def main():
    split_image("image_from_dataset.png")


if __name__ == "__main__":
    main()
