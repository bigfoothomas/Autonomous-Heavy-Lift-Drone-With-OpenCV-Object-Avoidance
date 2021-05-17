import cv2
import numpy as np

# Load intrinsic and extrensic camera values
data_cam1 = np.load('cam1.npz')
mtx = data_cam1['mtx.npy']
dist = data_cam1['dist.npy']
imgpoints = data_cam1['imgpoints.npy']
gray = data_cam1['gray.npy']
data_cam2 = np.load('cam2.npz')
mtx2 = data_cam2['mtx2.npy']
dist2 = data_cam2['dist2.npy']
imgpoints2 = data_cam2['imgpoints2.npy']
data_stereo_cal = np.load('stereo_cal.npz')
R = data_stereo_cal['R.npy']
T = data_stereo_cal['T.npy'] 


# Computes rectification transforms
flag = cv2.CALIB_ZERO_DISPARITY
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx, dist, mtx2, dist2, gray.shape[::-1], R, T, alpha = -0.5)

# Computes the undistortion and rectification transformation map to be used by the remap function
mapX1, mapY1 = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, gray.shape[::-1], cv2.CV_32FC1) 
mapX2, mapY2 = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, gray.shape[::-1], cv2.CV_32FC1)

cap1 = cv2.VideoCapture(0)
cap1.set(4,360)
cap1.set(3,480)
cap2 = cv2.VideoCapture(2)
cap2.set(4,360)
cap2.set(3,480)
MinContourArea = 4000
counter = 0

# Write live video to file 
#w = cap1.get(cv2.CAP_PROP_FRAME_WIDTH);
#h = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT);
#writer = cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (int(w),int(h)))

while True:
    
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()
    
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Find frames per second and dislay it on screen
    timer = cv2.getTickCount()
    fps = ((cv2.getTickFrequency()/(cv2.getTickCount() - timer))*0.0001)
    cv2.putText(frame1, 'fps = ' + str(int(fps)),(0,20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    
    # Finds interpolated pixel value from gray_frame1 that relates to new cap1_rectified
    cap1_rectified = cv2.remap(gray_frame1, mapX1, mapY1, cv2.INTER_LINEAR) 
    cap2_rectified = cv2.remap(gray_frame2, mapX2, mapY2, cv2.INTER_LINEAR)
    
    # Initializes disparity map
    retval1 = cv2.StereoBM_create(numDisparities=96, blockSize=19) 
    
    # Computes raw disparity map
    disparity_map = retval1.compute(cap1_rectified, cap2_rectified) 
    rotate_disparity = cv2.rotate(disparity_map, cv2.ROTATE_180) 
    
    # Adjusts pixel intensity to create a clearer image
    convert = cv2.normalize(rotate_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Converts values to binary
    ret, thresh = cv2.threshold(convert,127,255,cv2.THRESH_BINARY)
    
     # Dilate, blur, and find contours
    blur = cv2.medianBlur(convert, 1)
    kernel = np.ones((3,3),np.uint8)
    dilate = cv2.dilate(blur, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        
        if cv2.contourArea(c) < MinContourArea:
            continue
        
        # Generates a bounding rectangle for a contour area bigger than MinContourArea
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Generates the object's centroid
        CoordXCentroid = (x+x+w)//2
        CoordYCentroid = (y+y+h)//2
        ObjectCentroid = (CoordXCentroid,CoordYCentroid)
        cv2.circle(frame1, ObjectCentroid, 1, (0, 0, 0), 5)
       
        # Display concept behind avoidance algorithm. Screen is split into right and left sides with message displayed depending on location of centroid.  
        if CoordXCentroid < 320:
            cv2.putText(frame1, 'TURN RIGHT',(150,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        elif CoordXCentroid >= 320:
            cv2.putText(frame1, 'TURN LEFT',(0,200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    
    cv2.imshow('Left', frame1) 
    cv2.imshow('Right', dilate) 
    #writer.write(frame1)
    k = cv2.waitKey(32)
    
    # Save images from video by pressing 't' and break with spacebar.
    if k == ord('t'): 
         
         cv2.imwrite('/home/pi/Pictures/test_images/object_detection{:1}.jpg'.format(counter), frame1)
         cv2.imwrite('/home/pi/Pictures/test_images/disp{:1}.jpg'.format(counter), convert)
         counter += 1
    
    elif k == 32:
        break

cap1.release()
cap2.release()
#writer.release()
cv2.destroyAllWindows()