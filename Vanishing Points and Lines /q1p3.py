import math
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation


def draw_lines(lines, result_image, resultName):
    length = 20000
    for i in range(len(lines)):
        r = lines[i][0][0]
        theta = lines[i][0][1]
        pt1 = (int(r * math.cos(theta) + length * math.sin(theta)), int(r * math.sin(theta) - length * math.cos(theta)))
        pt2 = (int(r * math.cos(theta) - length * math.sin(theta)), int(r * math.sin(theta) + length * math.cos(theta)))
        cv.line(result_image, pt1, pt2, color=(0, 0, 255), thickness=10)

    # save_image(result_image, resultName)


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


def filter_gap(lines, distance_threshold):
    drawed = [True for i in range(len(lines))]
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            r1 = lines[i][0][0]
            r2 = lines[j][0][0]
            if abs(r1 - r2) < distance_threshold:
                drawed[j] = False

    new_lines = []
    for i in range(len(lines)):
        if drawed[i]:
            new_lines.append(lines[i])

    return new_lines


def find_intersection(lines):
    length = 10000
    a = np.zeros((len(lines), 2), dtype="float64")
    b = np.zeros((len(lines), 1), dtype="float64")
    counter = 0
    for line in lines:
        theta = line[0][1]
        r = line[0][0]
        pt1 = (int(r * math.cos(theta) + length * math.sin(theta)), int(r * math.sin(theta) - length * math.cos(theta)))
        pt2 = (int(r * math.cos(theta) - length * math.sin(theta)), int(r * math.sin(theta) + length * math.cos(theta)))
        slope = float((pt2[1] - pt1[1])) / float((pt2[0] - pt1[0]))
        intercept = pt2[1] - slope * pt2[0]
        a[counter, 0] = -slope
        a[counter, 1] = 1
        b[counter, 0] = intercept
        counter = counter + 1

    x = np.linalg.lstsq(a, b, rcond=None)
    point = x[0]
    point = np.array([int(point[0, 0]), int(point[1, 0]), 1])
    return point


def find_calibration_matrix(vx_point, vy_point, vz_point):
    A = np.array([[vx_point[0] - vz_point[0], vx_point[1] - vz_point[1]],
                  [vy_point[0] - vz_point[0], vy_point[1] - vz_point[1]]], dtype="float64")

    b = np.array([[vy_point[0] * (vx_point[0] - vz_point[0]) + vy_point[1] * (vx_point[1] - vz_point[1])],
                  [vx_point[0] * (vy_point[0] - vz_point[0]) + vx_point[1] * (vy_point[1] - vz_point[1])]],
                 dtype="float64")

    principal_pont = np.linalg.solve(A, b)
    px = principal_pont[0, 0]
    py = principal_pont[1, 0]

    f2 = -px * px - py * py + (vx_point[0] + vy_point[0]) * px + (vx_point[1] + vy_point[1]) * py - (
            vx_point[0] * vy_point[0] + vx_point[1] * vy_point[1])


    f = math.sqrt(f2)
    k = np.array([[f, 0, px],
                  [0, f, py],
                  [0, 0, 1]])

    return k


def find_horizantal_line(vx, vy):
    h = np.cross(vx, vy)
    h = h.astype("float64")
    # print(h)
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
    return h, a, b, c


def find_size_of_total_frame(frame,homography):
    points = []
    src_points = np.array([[0, 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]],
                           [frame.shape[1], 0]], dtype="float32")


    dst = cv.perspectiveTransform(src_points.reshape(-1, 1, 2),homography)
    dst = dst.reshape(src_points.shape)
    # print(dst.shape)
    # print(dst)
    points.append(dst[0])
    points.append(dst[1])
    points.append(dst[2])
    points.append(dst[3])
    min = np.min(points, axis=0)
    max = np.max(points, axis=0)
    size = max - min
    size = size.astype("int64")
    translation = np.array([[1, 0, -min[0]], [0, 1, -min[1]], [0, 0, 1]])
    return translation, size


image = cv.imread("vns.jpg")
image_blur = cv.GaussianBlur(image, (5, 5), 0)
gray = cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)
kernel = np.ones((11, 11), np.uint8)
gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
edges = cv.Canny(gray, 100, 500)
# save_image(edges, "edge")
x_direction = cv.HoughLines(edges, 1, np.pi / 180, 100, min_theta=np.pi / 2, max_theta=3 * np.pi / 4)
x_direction = filter_gap(x_direction, 100)
x_lines_image = np.copy(image)
draw_lines(x_direction, x_lines_image, "xLines")

y_direction = cv.HoughLines(edges, 1, np.pi / 180, 130, min_theta=np.pi / 4, max_theta=np.pi / 2)
y_direction = filter_gap(y_direction, 80)
y_lines_image = np.copy(image)
draw_lines(y_direction, y_lines_image, "yLines")

z_direction = cv.HoughLines(edges, 1, np.pi / 180, 100, min_theta=9 * np.pi / 10)
z_direction = filter_gap(z_direction, 80)
z_lines_image = np.copy(image)
draw_lines(z_direction, z_lines_image, "zLines")

x_van_point = find_intersection(x_direction)
y_van_point = find_intersection(y_direction)
z_van_point = find_intersection(z_direction)

k = find_calibration_matrix(x_van_point, y_van_point, z_van_point)
# image_for_principal_point = np.copy(image)
# cv.drawMarker(image_for_principal_point, (int(k[0, 2]), int(k[1, 2])), color=(0, 0, 255), thickness=50)
# image_for_principal_point = cv.cvtColor(image_for_principal_point, cv.COLOR_BGR2RGB)
# plt.imshow(image_for_principal_point)
# plt.title(f"focal length: {k[0, 0]}")
# plt.savefig("res03.jpg")

k_inv = np.linalg.inv(k)
vx_back = np.matmul(k_inv, x_van_point)
vy_back = np.matmul(k_inv, y_van_point)
vz_back = np.matmul(k_inv, z_van_point)

h, a, b, c = find_horizantal_line(x_van_point, y_van_point)
slope = -h[0] / h[1]
angle_z = math.atan(slope)
# angle_z=np.rad2deg(angle_z)
print(angle_z)

z_axis = np.array([0, 0, 1])
l = np.cross(vx_back, vy_back)
angle_x = math.acos(np.dot(z_axis, l) / (np.linalg.norm(z_axis) * np.linalg.norm(l)))
angle_x = np.pi / 2 - angle_x
# angle_x=np.rad2deg(angle_x)
print(angle_x)

R_z = Rotation.from_euler('z', -angle_z).as_matrix()
R_x = Rotation.from_euler('x', angle_x).as_matrix()
h = np.matmul(k, np.matmul(R_x, np.matmul(R_z, k_inv)))
translation,size=find_size_of_total_frame(image,h)
result = cv.warpPerspective(image, np.matmul(translation,h), (size[0],size[1]))
save_image(result, "res04")
