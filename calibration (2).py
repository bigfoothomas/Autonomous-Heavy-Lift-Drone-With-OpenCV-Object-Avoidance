import cv2
import numpy as np
import glob
import os

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2) 
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane
imgpoints2 = []

images = sorted(glob.glob('/home/pi/Pictures/Cam1Rough/*.jpg'))
images2 = sorted(glob.glob('/home/pi/Pictures/Cam2Rough/*.jpg'))

# Counter for image points. Image points from each camera must be equal for cv2.stereoCalibrate.
counter1 = 0 
counter2 = 0

for fname in images:
    img = cv2.imread(fname) 
    
    # Convert from color to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    
    # Find the corners of the chessboard pattern
    ret, corners = cv2.findChessboardCorners(gray,(6,8),None) 
    if ret == True:
        objpoints.append(objp)
        
        # Narrow image points down to subpixel level for more accuracy
        corners1 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria) 
        imgpoints.append(corners1)
        
        # Connect corners with lines to provide a visual aid for calibration 
        img = cv2.drawChessboardCorners(img,(6,8),corners1,ret) 
        cv2.imwrite('/home/pi/Pictures/Cam1Corners/cam1c{:1}.jpg'.format(counter1), img)  
        counter1 += 1
        cv2.imshow('img',img)
        cv2.waitKey(500)
    else:
        
        # Remove any image where chessboard corners cannot be located
        os.remove(fname)

# Repeat the same process for the second camera.
for fname2 in images2:
    img2 = cv2.imread(fname2)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret2, corners2 = cv2.findChessboardCorners(gray2,(6,8),None)
    if ret2 == True:
        corners3 = cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
        imgpoints2.append(corners3)
        img2 = cv2.drawChessboardCorners(img2,(6,8),corners3,ret2)
        cv2.imwrite('/home/pi/Pictures/Cam2Corners/cam2c{:1}.jpg'.format(counter2), img2)
        counter2 += 1
        cv2.imshow('img2',img2)
        cv2.waitKey(32)
    else:
        os.remove(fname2)

# Utilize cv2.calibrateCamera to find camera's instrinsic values
ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
ret2,mtx2,dist2,rvecs2,tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)

# Save each camera's returned values 
np.savez('cam1.npz', mtx=mtx, dist=dist, imgpoints=imgpoints, gray=gray)
np.savez('cam2.npz', mtx2=mtx2, dist2=dist2, imgpoints2=imgpoints2)

# Fix camera's instrinsic values that were previously returned and fix focal length. Utilize cv2.stereoCalibrate to find extrinsic values for the camera pair
criteria2 = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
 
flag = cv2.CALIB_FIX_INTRINSIC,cv2.CALIB_SAME_FOCAL_LENGTH, cv2.CALIB_ZERO_TANGENT_DIST 
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints, imgpoints2, mtx, dist, mtx2, dist2, gray.shape[::-1], criteria2, flag)

# Save extrinsic values
np.savez('stereo_cal.npz', R=R, T=T)

