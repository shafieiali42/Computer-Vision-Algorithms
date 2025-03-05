import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def draw_lines(lines,result_image,resultName):
    length=20000
    for i in range(len(lines)):
        r = lines[i][0][0]
        theta = lines[i][0][1]
        pt1 = (int(r * math.cos(theta) + length * math.sin(theta)), int(r * math.sin(theta) - length * math.cos(theta)))
        pt2 = (int(r * math.cos(theta) - length * math.sin(theta)), int(r * math.sin(theta) + length * math.cos(theta)))
        cv.line(result_image, pt1, pt2, color=(0, 0, 255), thickness=10)

    # save_image(result_image,resultName)


def my_clip(image, cast):
    image = image.astype("float64")
    if np.min(image) == np.max(image):
        if np.ndim(image) == 3:
            image[:, :, :] = 0
        else:
            image[:, :] = 0
    image = image - np.min(image)
    image = (255 / (np.max(image) - np.min(image))) * image
    if cast:
        image = image.astype("uint8")
    return image


def save_image(image, name):
    image = my_clip(image, True)
    image = image.astype("uint8")
    cv.imwrite(f"{name}.jpg", image)


def filter_gap(lines,distance_threshold):
    drawed=[True for i in range(len(lines))]
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            r1=lines[i][0][0]
            r2=lines[j][0][0]
            if abs(r1 - r2) < distance_threshold:
                drawed[j] = False

    new_lines=[]
    for i in range(len(lines)):
        if drawed[i]:
            new_lines.append(lines[i])

    return new_lines



def find_intersection(lines):
    length=10000
    a=np.zeros((len(lines),2),dtype="float64")
    b=np.zeros((len(lines),1),dtype="float64")
    counter=0
    for line in lines:
        theta = line[0][1]
        r = line[0][0]
        pt1 = (int(r * math.cos(theta) + length * math.sin(theta)), int(r * math.sin(theta) - length * math.cos(theta)))
        pt2 = (int(r * math.cos(theta) - length * math.sin(theta)), int(r * math.sin(theta) + length * math.cos(theta)))
        slope=float((pt2[1]-pt1[1]))/float((pt2[0]-pt1[0]))
        intercept=pt2[1]-slope*pt2[0]
        a[counter,0]=-slope
        a[counter,1]=1
        b[counter,0]=intercept
        counter=counter+1

    x=np.linalg.lstsq(a,b,rcond=None)
    point=x[0]
    # print(point.shape)
    # print(point)
    point=np.array([int(point[0,0]),int(point[1,0]),1])
    print(point)
    # print(point)
    return point


image=cv.imread("vns.jpg")
image_blur=cv.GaussianBlur(image,(5,5),0)
gray = cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)
kernel = np.ones((11, 11), np.uint8)
gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
edges = cv.Canny(gray, 100, 500)
# save_image(edges,"edge")
x_direction=cv.HoughLines(edges,1,np.pi/180,100,min_theta=np.pi/2,max_theta=3*np.pi/4)
x_direction=filter_gap(x_direction,100)
x_lines_image=np.copy(image)
draw_lines(x_direction,x_lines_image,"xLines")

y_direction=cv.HoughLines(edges,1,np.pi/180,130,min_theta=np.pi/4,max_theta=np.pi/2)
y_direction=filter_gap(y_direction,80)
y_lines_image=np.copy(image)
draw_lines(y_direction,y_lines_image,"yLines")


z_direction=cv.HoughLines(edges,1,np.pi/180,100,min_theta=9*np.pi/10)
z_direction=filter_gap(z_direction,80)
z_lines_image=np.copy(image)
draw_lines(z_direction,z_lines_image,"zLines")



x_van_point=find_intersection(x_direction)
y_van_point=find_intersection(y_direction)
z_van_point=find_intersection(z_direction)



def find_horizantal_line(vx,vy):
    h = np.cross(vx, vy)
    h = h.astype("float64")
    print(h)
    a = h[0]
    b = h[1]
    c = h[2]
    a2 = a / math.sqrt(a * a + b * b)
    b2 = b / math.sqrt(a * a + b * b)
    c2 = c / math.sqrt(a * a + b * b)
    h[0] = a2
    h[1] = b2
    h[2] = c2
    a = a2
    b = b2
    c = c2
    return h,a,b,c


h,a,b,c=find_horizantal_line(x_van_point,y_van_point)
print(f"a: {a}")
print(f"b: {b}")
print(f"c: {c}")
first_point_on_h=(0,int(-c/b))
second_point_on_h=(int(-c/a),0)
image_for_h=np.zeros((5000,5000,3))
image_for_h[:image.shape[0],:image.shape[1],:]=image
cv.line(image_for_h,first_point_on_h,second_point_on_h,color=(0,0,255),thickness=20)
save_image(image_for_h,"res01")

plt.imshow(image)
plt.scatter(x_van_point[0],x_van_point[1])
plt.scatter(y_van_point[0],y_van_point[1])
plt.scatter(z_van_point[0],z_van_point[1])
plt.plot([x_van_point[0],y_van_point[0]],[x_van_point[1],y_van_point[1]])
plt.savefig("res02.jpg")


