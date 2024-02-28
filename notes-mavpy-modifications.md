# Documentation on how to add a button to take pictures from the camera GUI


### Mav.py
```python
# ---------------------------- Take Picture Method ---------------------------- #

import cv2


# Line 648 in mav.py
def takePicture(self):
    """
    This method takes a picture from the camera and saves it to the user
    selected folder
    """
    images = self.cam.daq()
    directory = "calibration_images"
    if images != None:  # Only accessed if camera is webcam
        print(f"Saving left image as => {images[0]['name']}_{self.takePictureCounter}")
        try:
            cv2.imwrite(
                f"{directory}/{images[0]['name']}_{self.takePictureCounter}.png",
                images[0]["img"],
            )
            print(f"Image saved as => {images[0]['name']}")
        except Exception as e:
            print(f"Error saving left image => {e}")

        print(f"Saving right image as => {images[1]['name']}_{self.takePictureCounter}")
        try:
            cv2.imwrite(
                f"{directory}/{images[1]['name']}_{self.takePictureCounter}.png",
                images[1]["img"],
            )
            print(f"Image saved as => {images[1]['name']}")
        except Exception as e:
            print(f"Error saving right image => {e}")
        self.takePictureCounter += 1
```
### kvSubPanels/camctrls.kv
```
# ---------------------------- Take Picture Button ---------------------------- #

"""
Label:
  text: '[b]Take Picture:[/b]'
  markup: True
  halign: 'right'
  valign: 'middle'
  text_size: self.size

Button:
  text: 'Click'
  size_hint_x: None
  width: 100
  on_press: root.parent.takePicture()

"""
```
