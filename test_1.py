import numpy as np
import cv2
import apriltag
from matplotlib import pyplot as plt
import open3d
import sys
from datetime import datetime
import os

# camera intrinsics
#[[834.27154541   0.         312.73457454]
# [  0.         830.28051758 264.76828241]
# [  0.           0.           1.        ]]

# imagepath = 'test_image_2.jpg'
# image = cv2.imread(imagepath)
# image = cv2.resize(image, (504, 672))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ply_path = "/data_c/ycb_data/data/035_power_drill/clouds/merged_cloud.ply"
pcd = open3d.io.read_point_cloud(ply_path)
downpcd = open3d.voxel_down_sample(pcd, voxel_size = 0.005)
xyz = np.asarray(downpcd.points)
rgb = np.asarray(downpcd.colors)

camera_params = np.array([834.27154541, 830.28051758, 312.73457454, 264.76828241])
fx = 834.27154541;
fy = 830.28051758;
cx = 312.73457454;
cy = 264.76828241;
intrinsic_matrix = np.array([[fx, 0 ,cx], [0, fy, cy], [0, 0, 1]])
tag_size = 0.028
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (640,480))

now = datetime.now()
LOG_DIR = "poses/" + now.strftime("%Y%m%d-%H%M%S") + "/"
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

while(True):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = apriltag.Detector()
    detections = detector.detect(gray)

    for r in detections:
        pose, e0, e1 = detector.detection_pose(r,camera_params,tag_size)
        R = pose[0:3,0:3]
        t = pose[0:3,3]

        with open(LOG_DIR + "rotation.txt", 'ab') as f:
            np.savetxt(f, np.reshape(R, (-1)), fmt='%.4f', delimiter=',', newline=' ')
            f.write(b'\n')
        with open(LOG_DIR + "translation.txt", 'ab') as f:
            np.savetxt(f, t, fmt='%.4f', delimiter=',', newline=' ')
            f.write(b'\n')

        xyz_rot = np.dot(np.matmul(np.array([[1,0,0],[0,-1,0],[0,0,-1]]),R), xyz.T)
        xyz_rot_tran = np.add(xyz_rot,np.expand_dims(t,1)).T*10

        # object point to image point
        point_rotation = np.identity(3)
        point_pose = np.hstack([point_rotation, np.expand_dims(t,1)])
        point_3D_homogeneous = np.hstack([xyz_rot_tran, np.ones((np.shape(xyz_rot_tran)[0],1))])
        point_transformation = np.matmul(intrinsic_matrix, point_pose)
        point_2D_homogeneous = np.matmul(point_transformation, point_3D_homogeneous.T)
        point_2D = point_2D_homogeneous[0:2,:] / point_2D_homogeneous[2,:]

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print("[INFO] tag family: {}".format(tagFamily))
        for point, color in zip(point_2D.T, rgb):
            cv2.circle(image, (int(point[0]), int(point[1])), 1, (int(color[2]*255), int(color[1]*255), int(color[0]*255)))

    # show the output image after AprilTag detection
    cv2.imshow("Image", image)
    out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
