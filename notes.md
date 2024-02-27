# Personal Notes on how the miniAV repository handles cameras

### Generic Camera Class for the API
```python
class camAPI(object):
  """
    'Parent' class use to act as generic API for all cameras
  """
  def __init__(self):
    """
      Camera properties
    """
    self.src = None          # Object Place holder for a camera object
    self.camOn = False       # Flag to indicate if the camera is active or not
    self.imgHeight = 0       # Integer number of pixels
    self.imgWidth = 0        # Integer number of pixels
    self.imgColorDepth = 3   # Number of color matrices. Default is 3 for RGB
    self.imgNumPixels = 0
    self.frameRate = 0       # Integer indicating the user desired frame rate
    self.rtRate = 0          # Measured FPS to get an image in real time
    self.name = ''           # String containing the name of the currently selected/active camera
    self.camList = []        # List of currently available on the system based on the SDK
                             # avalability
    if zSDK is not None:
      self.camList.append('ZED')
    if rsSDK is not None:
      self.camList.append('Intel RS')
    # Finally assume that openCV is installed
    self.camList.append('Web Cam')


  def activateCam(self, camModel):
    if camModel == 'ZED' and zSDK is not None:
      self.name = 'ZED'
      self.src = zedCam(self)
    elif camModel == 'Intel RS' and rsSDK is not None:
      self.name = 'Intel RS'
      self.src = rsCam(self)
    elif camModel == 'Web Cam' or camModel == 'Intel RS':
      self.name = 'Web Cam'
      self.src = webCam(self)
    else:
      raise ValueError(f'Unsupported Camera Model => {camModel}',\
                        'method: activateCam', \
                        ' class: camAPI')


  def daq(self, flip=False):
    """
      Method that gets the image from the camera source
    """
    # Make the camera is ON before acquiring
    if self.camOn:
      # Note: the getImage method in all sub-classes return a list of image
      #       because some camera can return more than 1 image depending on the
      #       settings.
      return self.src.getImage(flip)

```

### WebCam python class

```python
class webCam(object):
  """
    Class to handle a generic web camera
  """
  def __init__(self, refAPI):
    """
      Initialize the web camera properties
    """ 
    self.api = refAPI
    self.camIdLeft = 1
    self.camIdRight = 0
    self.camLeft = cv2.VideoCapture(self.camIdLeft, cv2.CAP_DSHOW)
    self.camRight = cv2.VideoCapture(self.camIdRight, cv2.CAP_DSHOW)
 
 
  def initCam(self):
    """
      Method that initialize the camera acquisition
    """
    if self.api.frameRate != 30:
      self.api.frameRate = 30
    # Note: For some reason the Logitech webcam yield the best daqLoop FPS when
    #       it is set to 30 FPS. Any number lower or higher than 30 for the
    #       webcam decreases the overall daqLoop FPS???
    self.camLeft = cv2.VideoCapture(self.camIdLeft, cv2.CAP_DSHOW)
    self.camLeft.set(cv2.CAP_PROP_FPS, self.api.frameRate)
    self.camLeft.set(cv2.CAP_PROP_FRAME_WIDTH, self.api.imgWidth)
    self.camLeft.set(cv2.CAP_PROP_FRAME_HEIGHT, self.api.imgHeight)
    if self.camLeft.isOpened(): # try to get the first frame
      self.api.camOn, _ = self.camLeft.read()
    else:
      raise ValueError(f'Left Webcam cannot be opened??', \
                        'method: initCam', \
                        ' class: webCam')

    self.camRight = cv2.VideoCapture(self.camIdRight, cv2.CAP_DSHOW)
    self.camRight.set(cv2.CAP_PROP_FPS, self.api.frameRate)
    self.camRight.set(cv2.CAP_PROP_FRAME_WIDTH, self.api.imgWidth)
    self.camRight.set(cv2.CAP_PROP_FRAME_HEIGHT, self.api.imgHeight)
    if self.camRight.isOpened(): # try to get the first frame
      self.api.camOn, _ = self.camRight.read()
    else:
      raise ValueError(f'Right Webcam cannot be opened??', \
                        'method: initCam', \
                        ' class: webCam')


  def getImage(self, flip=False):
    """
      Method that retrieves a new image from a web camera
    """
    tStart = time.time()   
    retLeft, newImgLeft = self.camLeft.read()
    retRight, newImgRight = self.camRight.read()
    self.api.rtRate = 1/((time.time() - tStart) + 1e-8) # Add epsilon to make
                                                        # sure we do not get
                                                        # divide by 0
    if retLeft and retRight:
      self.api.camOn = True 
      # Process the image to display to the user
      newImgLeft = np.flipud(newImgLeft)
      newImgRight = np.flipud(newImgRight)

      if flip:
        newImgLeft, newImgRight = newImgRight, newImgLeft
      # Re-order the colors to get the correct color image and send it back as
      # a list of dict to be compatible with all the getImage method in the API
      # (some camera can return more than 1 img)
      return [{'name': 'imgLeft', 'img': newImgLeft[:,:,[2,1,0]]}, {'name': 'imgRight', 'img': newImgRight[:,:,[2,1,0]]}]
    else:
      self.api.camOn = False 


  def close(self):
    self.camLeft.release()  
    self.camRight.release()  
```

### Generic Camera Class for the API

```python

    #-------------------------- CAMERA properties -----------------------------#
    self.cam = camAPI()        # camAPI object
    # Image mode => the spinner method will set the camera source and image
    #               type plus initialize the TEXTURE accordingly using 
    #               "resetWin()" method called by the "spinnerSelectCam()"
    #                method
    self.imgResName = self.app.uiUtils.appsGetDflt('imgResName')
    if self.imgResName is None:
      self.imgResName = 'VGA'
    self.camCtrls.camRes.text = self.imgResName
    self.spinnerCamRes(self.imgResName)
    # update the spinner cam list based on the available SDK
    self.camCtrls.selectCam.values = self.cam.camList
    self.cam.name = self.app.uiUtils.appsGetDflt('camName')
    if self.cam.name is None:
      self.cam.name = 'Web Cam'
    self.camCtrls.selectCam.text = self.cam.name  
    self.spinnerSelectCam(self.cam.name)
    # Remote Type => this will set the remote type currently active 
    self.app.remoteType = self.app.uiUtils.appsGetDflt('remoteType')
    if self.app.remoteType not in ['Tactic3ch', 'TQi4ch']:
      self.app.remoteType = 'Tactic3ch'
    self.spinnerRemoteType(self.app.remoteType)
    self.camCtrls.remType.text = self.app.remoteType
     # ZED camera frame rate - the spinner method will set the TEXTURE
     # accordingly using "resetWin" method   
    self.cam.frameRate = self.app.uiUtils.appsGetDflt('camFrameRate')
    if self.cam.frameRate is None:
      self.cam.frameRate = 30
    self.camCtrls.frameRate.text = str(self.cam.frameRate)      
    self.spinnerFrameRate(self.camCtrls.frameRate.text)
```
