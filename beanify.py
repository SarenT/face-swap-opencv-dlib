#! /usr/bin/env python

import argparse
import sys
import numpy as np
import cv2
from imutils import face_utils
import datetime
import imutils
import time
import dlib
import os
import secrets
from utils import face_swap, face_swap2, face_swap3, niceTime
import subprocess

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description='Process webcam feed in realtime to swap faces with other reference images')
ap.add_argument("-v", "--verbose", required=False, default="False", help="Verbose output")
ap.add_argument("-b", "--benchmark", required=False, default="False", help="Print out how long each step requires in milliseconds")
ap.add_argument('-d', "--device", required=True, help="Path to loopback device")
ap.add_argument('-w', "--webcam", required=True, help="Path to actual webcam device")
#
args = vars(ap.parse_args())
print(args)

benchmark = False
verbose = False

benchmark = args['benchmark'] == "True" or args['benchmark'] == "1"
verbose = args['verbose'] == "True" or args['verbose'] == "1"
device = args['device']
webcam = args['webcam']

# Make sure OpenCV is version 3.0 or above
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# FFMPEG Gommands:
# Play in a loop:
# ffmpeg -stream_loop -1 -re -i Desktop/Misc/example.mp4 -map 0:v -f v4l2 -s 640x480 /dev/video3
# Play a file:
# ffmpeg -re -i Desktop/Misc/example.mp4 -map 0:v -f v4l2 -s 640x480 /dev/video3
# Playing from fifo (webcam_fifo)
# ffmpeg -f rawvideo -s 640x480 -pix_fmt bgr24 -i webcam_fifo -map 0:v -f v4l2 -s 640x480 /dev/video3


if int(major_ver) < 3 :
	print >>sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
	sys.exit(1)

print("[INFO] loading facial landmark predictor...")
model = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

imgDir = '/home/stasciya/git/face-swap-opencv-dlib/imgs/'
rotateImagesDir = imgDir + 'rotate/'
imgNames = []
for file in os.listdir(rotateImagesDir):
	if file.endswith(".jpg"):
		imgNames.append(file)
imgInd = 0

beanImagePath = imgDir + 'MrBeanTransparent_WhiteBG2.jpg'


print(beanImagePath)
beanMode = False
modeChange = True

process = subprocess.Popen(["ffmpeg", "-f", "rawvideo", "-s", "640x360", "-pix_fmt", "bgr24", "-i", "pipe:", "-map", "0:v", "-f", "v4l2", "-pix_fmt", "yuv420p", "-s", "640x360", device], stdin=subprocess.PIPE)
feedStream = process.stdin

print("Starting capture...")
video_capture = cv2.VideoCapture(webcam)
print("Capturing...")
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

print("Output open")

showLines = False
showLabel = False
label = None
blank = False
imgSize = None
blank_image = None
nowWholeLoop = time.time()
now = time.time()
while True:
	if modeChange:
		if beanMode:
			faceImage = cv2.imread(beanImagePath)
			noseShift = [180, 0]
		else:
			if verbose : print("Showing", imgNames[imgInd], 'at', rotateImagesDir + imgNames[imgInd])
			
			faceImage = cv2.imread(rotateImagesDir + imgNames[imgInd])
			noseShift = [0, 0]
			
			if benchmark : print("\t", "Mode change read", niceTime(now)); now = time.time()
	if blank and blank_image is None:
		if verbose : print(imgSize)
		
		blank_image = np.zeros((imgSize[0],imgSize[1],3), np.uint8)
		
		if benchmark : print("\t", "Blank", niceTime(now)); now = time.time()
		
	if not blank:
		ret, img2 = video_capture.read()
		
		if benchmark : print("\t", "Video capture", niceTime(now)); now = time.time()
		
		imgSize = img2.shape
		gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		# detect faces in the grayscale frame
		rects2 = detector(gray2, 0)
		
		if benchmark : print("\t", "detector", niceTime(now)); now = time.time()
	
	if len(rects2) != 0 or blank:
		#print("|", end='')
		try:
			if blank:
				feedStream.write(blank_image.tobytes())
				
				if benchmark : print("\t", "Blank write", niceTime(now)); now = time.time()
				
				output2 = blank_image
				points1 = []
			else:
				output1, output2, points1 = face_swap2(img2, faceImage, detector, predictor, noseShift, beanMode = beanMode, modeChange = modeChange, rects1 = rects2, gray1 = gray2, benchmark = benchmark, verbose = verbose)
				
				if benchmark : print("\t", "Face swap", niceTime(now)); now = time.time()
			if showLines:
				output2 = cv2.polylines(output2, np.int32([points1]), True, (0, 255, 0))
			
			if modeChange:
				if label is not None:
					cv2.destroyWindow(label)
					
				if beanMode:
					label = "Mr. Bean: Whistler's Mother"
				elif blank:
					label = "Blank"
				else:
					label = os.path.splitext(imgNames[imgInd])[0]
			
			if showLabel:
				#print(imgSize)
				output2 = cv2.putText(output2, label, (10, imgSize[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
			#feed = cv2.cvtColor(output2, cv2.COLOR_BGR2YUV_YV12)
			if output2 is not None:
				cv2.imshow(label, output2)
				
				if benchmark : print("\t", "Image show", niceTime(now)); now = time.time()
				feedStream.write(output2.tobytes())
				if benchmark : print("\t", "Image write", niceTime(now)); now = time.time()
			else:
				cv2.imshow(label, img2)
				
				if benchmark : print("\t", "Image show failed", niceTime(now)); now = time.time()
			cv2.moveWindow(label, 0, 0)
		    
		except cv2.error as e:
		    if verbose : print(e)
			
	else:
		if verbose : print(".", end='')
	
	if modeChange:
		modeChange = False
	code = cv2.waitKey(1)
	if code & 0xFF == ord('q'):
		video_capture.release()
		feedStream.flush()
		feedStream.close()
		process.terminate()
		cv2.destroyAllWindows()
		sys.exit(0)
		break
	else:
		if code == ord('w'):
			noseShift[1] += 10
		elif code == ord('a'):
			noseShift[0] += 10
		elif code == ord('s'):
			noseShift[1] -= 10
		elif code == ord('d'):
			noseShift[0] -= 10
		elif code == ord('z'):
			beanMode = False
			imgInd -= 1
			imgInd %= len(imgNames)
			modeChange = True
		elif code == ord('x'):
			beanMode = False
			imgInd += 1
			imgInd %= len(imgNames)
			modeChange = True
		elif code == ord('b'):
			beanMode = True
			modeChange = True
		elif code == ord('e'):
			showLines = not showLines
		elif code == ord('l'):
			showLabel = not showLabel
			if verbose : print("showLabel", showLabel)
		elif code == ord('k'):
			blank = not blank
			if blank:
				video_capture.release()
			else:
				video_capture.open(0)
				video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
				video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
			modeChange = True
		if code > -1:
			if verbose : print(code)
			if verbose : print(noseShift)
		
	if benchmark : print("One loop", niceTime(nowWholeLoop)); nowWholeLoop = time.time()
	
	#time.sleep(0.04)

	
	#output1, output2 = face_swap2(img2,faceImage, detector, predictor)
	#assert output1 is not None or output1 is not None, "There were not found faces in either the first or second image"
	#cv2.imshow("Face Swaqpped", output1)
	
	
	
	#cv2.imwrite(beanImagePath[:-4]+"_swapped.png",output1)
	#cv2.imwrite("imgs/webcam_swapped.png",output2)


#if not webcam:
#	if filename2 is None:
#		output = face_swap3(faceImage, detector, predictor)
#		assert output is not None, "There are not 2 faces or more in the image!"
#		cv2.imshow("Face Swapped", output)
#		cv2.imwrite(beanImagePath[:-4]+"_swapped.jpg",output)
#
#	else:
#		img2 = cv2.imread(filename2)
#		output1, output2 = face_swap2(img2,faceImage, detector, predictor)
#		assert output1 is not None or output1 is not None, "There were not found faces in either the first or second image"
#		print("Saving...")
#		cv2.imshow("Face Swapped", output1)
#		cv2.imwrite(beanImagePath[:-4]+"_swapped.png",output1)
#		cv2.imshow("Face Swapped2", output2)
#		cv2.imwrite(filename2[:-4]+"_swapped.png",output2)
#else:
		

