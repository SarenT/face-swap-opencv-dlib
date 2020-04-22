# Faceswapping using OpenCV and Dlib

Quick and simple app for face swapping using OpenCV, dlib and Tkinter from webcam feed into another webcam feed for video conferences
based on [this repository](https://github.com/charlielito/face-swap-opencv-dlib), which itself is based on [this respository](https://github.com/spmallick/learnopencv/tree/master/FaceSwap) for the core of the technique. More about on how it works in this [article](http://www.learnopencv.com/face-swap-using-opencv-c-python/).

## Requirements
* Python 2.7
* OpenCV 3.0+ with python bindings (needed to visualize the images/video)
* Numpy
* Pillow
* Tkinter
* Python bindings of dlib.
* v4l2loopback kernel module
* ffmpeg

#### Easy install
Build `OpenCV` or install the light version with `sudo apt-get install libopencv-dev python-opencv`. For Windows users it is always easier to just download the binaries of OpenCV and execute them, see [this web page](http://docs.opencv.org/trunk/d5/de5/tutorial_py_setup_in_windows.html). For `TKinter` in Linux just execute: `apt-get install python-tk` (python binaries in windows usually have Tkinter already installed).

For python dependences just execute:

```
pip install -r requirements.txt
```

***Dlib installation***: For Windows Users it can be very hard to compile dlib with python bindings, just follow this simple [instruction on how to install dlib in windows the easy way](https://github.com/charlielito/install-dlib-python-windows).

For Linux Users make sure you have the prerequisites:
```
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
```
Finally just `pip install it` with: `pip install dlib`

As for v4l2loopback, you can either use official repos or here [this repository](https://github.com/umlaeute/v4l2loopback/)

## Usage

First create a loopback device
```
# See what devices are already there: e.g. /dev/video0 /dev/video1
ls /dev/video*
# Now create a new one (once)
modprobe v4l2loopback devices= card_label="Loopback Camera" exclusive_caps=1
# Check the name of the new device e.g. /dev/video0 /dev/video1 */dev/video2*
ls /dev/video*

python3 beanify.py -d /dev/video2
```

You can enter change modes on the window or enter commands:
*k* ... blank feed and releases webcam
*q* ... quit application and release resources
*b* ... *beanify* aka enter **Mr. Bean: Whistler's Mother** mode.
*z*/*x* ... previous face/next face (in imgs/rotate folder)
*e* ... show face lines
*l* ... show image label
*wasd* ... adjust nose in beanify mode

Keep the face images small (pixel size) in the imgs/rotate folder for better performance.
