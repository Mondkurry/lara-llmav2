import tqdm as tqdm


class GUI:
    def __init__(self, num_images):
        self.num_images = num_images
        print(self.header())
        print(self.instructions())
        print(self.image_saved("path"))

    def header(self):
        return (
            f"|---------------------------------------------------|\n"
            f"|           MiniAV Camera Calibration Tool          |\n"
            f"|---------------------------------------------------|"
        )

    def instructions(self):
        return (
            f"Instructions: \n"
            f"1. Press 'q' to quit \n"
            f"2. Press 'n' to move to the next image \n"
            f"3. Press 'p' to move to the previous image \n"
            f"4. Press 'c' to clear images \n"
        )

    def image_saved(self, path):
        return f"| Image saved to {path} \n"

    def view_image(self, path):
        return f"Viewing image {path}"


def main():
    gui = GUI(10)


if __name__ == "__main__":
    main()
