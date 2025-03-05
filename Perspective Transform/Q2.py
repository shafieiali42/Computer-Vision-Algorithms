import math
import cv2 as cv
import numpy as np


def project_3d_point(calibration_matrix,rotation_matrix,translation_vector,point):
    P=np.zeros((3,4))
    P[:,:3]=np.copy(rotation_matrix)
    P[:,3]=np.copy(translation_vector)
    P=np.matmul(calibration_matrix,P)
    point=point.reshape(4,1)
    projection=np.matmul(P,point)
    if projection[2,0]!=0:
        projection=projection/projection[2,0]
    else:
        print("Devided by zero")
    projection=np.transpose(projection[:2,:])
    projection=projection.tolist()
    return projection


image = cv.imread("logo.png")





k1 = np.array([[500, 0, 2000],
               [0, 500,2000],
               [0, 0, 1]])

rotation_matrix1=np.identity(3)
translation_vector1=np.array([[0,0,0]])

k2 = np.array([[500, 0,128],
               [0, 500,128],
               [0, 0, 1]])

translation_vector2=np.array([[0, 40,0]])
theta=math.atan(-40/25)
rotation_matrix2=np.array([[1,0,0],
               [0,np.cos(theta),-np.sin(theta)],
               [0,np.sin(theta),np.cos(theta)]])


points=np.array([[64,128,1000,1],[64,-128,1000,1],[128,64,1000,1],[-128,64,1000,1]])


camera1_points=[]
camera2_points=[]
for i in range(points.shape[0]):
    camera1_points.append(project_3d_point(k1,rotation_matrix1,translation_vector1,points[i]))
    camera2_points.append(project_3d_point(k2,rotation_matrix2,translation_vector2,points[i]))

camera1_points=np.array(camera1_points)
camera2_points=np.array(camera2_points)
homography,mask=cv.findHomography(camera2_points,camera1_points)
image_size=4000
warped_image=cv.warpPerspective(image,homography,dsize=(image_size,image_size))
cv.imwrite("res12.jpg",warped_image)




