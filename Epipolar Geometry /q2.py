import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


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


image1 = cv.imread("01.JPG")
image2 = cv.imread("02.JPG")

sift = cv.SIFT.create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

image1_points = []
image2_points = []
for match in good_matches:
    image1_points.append(kp1[match.queryIdx].pt)
    image2_points.append(kp2[match.trainIdx].pt)

image1_points = np.array(image1_points, dtype="float64")
image2_points = np.array(image2_points, dtype="float64")
fundamental, mask = cv.findFundamentalMat(image1_points, image2_points, method=cv.FM_RANSAC, ransacReprojThreshold=3,
                                          confidence=0.995, maxIters=10000)

# concat = cv.drawMatches(image1, None, image2, None, None, None)
# save_image(concat, "concat")

# image_for_interest_points = np.copy(concat)
image1_interest_points = np.copy(image1)
image2_interest_points = np.copy(image2)

for counter, match in enumerate(good_matches):
    if mask[counter] == 1:
        cv.circle(image1_interest_points, (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1])), radius=5,
                  color=(0, 255, 0), thickness=-1)
        cv.circle(image2_interest_points, (int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1])), radius=5,
                  color=(0, 255, 0), thickness=-1)
    else:
        cv.circle(image1_interest_points, (int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1])), radius=5,
                  color=(0, 0, 255), thickness=-1)
        cv.circle(image2_interest_points, (int(kp2[match.trainIdx].pt[0]), int(kp2[match.trainIdx].pt[1])), radius=5,
                  color=(0, 0, 255), thickness=-1)

image_for_interest_points = np.concatenate((image1_interest_points, image2_interest_points), axis=1)
save_image(image_for_interest_points, "res05")

s, v, d = np.linalg.svd(fundamental)
h = d[-1, :]
h = h / h[2]
epipole1 = np.array([h[0], h[1]])

s, v, d = np.linalg.svd(np.transpose(fundamental))
h = d[-1, :]
h = h / h[2]
epipole2 = np.array([h[0], h[1]])

plt.imshow(image1)
plt.scatter(epipole1[0], epipole1[1])
plt.savefig("res06.jpg")
plt.clf()

plt.imshow(image2)
plt.scatter(epipole2[0], epipole2[1])
plt.savefig("res07.jpg")
plt.clf()


def find_epipolar_line(fundamental_matrix, point):
    point_copy = np.copy(point)
    point_copy = np.array([point_copy[0], point_copy[1], 1])
    line = np.matmul(fundamental_matrix, point_copy)
    return line


def draw_line(line, result,result2, color, pt):
    # print(line)
    # print()
    a = line[0]
    b = line[1]
    c = line[2]
    x0 = 0
    y0 = int(-c / b)
    x1 = result.shape[1]
    y1 = int((-c - a * x1) / b)
    # x1=int(-c/a)
    # y1=0
    # print((x0, y0))
    # print(x1, y1)
    # print()
    cv.line(result, (x0, y0), (x1, y1), color=color, thickness=5)
    cv.circle(result2, (int(pt[0]), int(pt[1])), radius=20, thickness=-1, color=color)


good_points1 = []
good_points2 = []
for i in range(image1_points.shape[0]):
    if mask[i]:
        good_points1.append(image1_points[i, :])
        good_points2.append(image2_points[i, :])

image1_for_lines = np.copy(image1)
image2_for_lines = np.copy(image2)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255),
          (0, 0, 0), (127, 64, 127), (64, 64, 255)]

for i in range(10):
    line1 = find_epipolar_line(fundamental, good_points1[i])
    line2 = find_epipolar_line(np.transpose(fundamental), good_points2[i])
    draw_line(line1, image2_for_lines,image1_for_lines, colors[i], good_points1[i])
    draw_line(line2, image1_for_lines,image2_for_lines, colors[i], good_points2[i])

# save_image(image1_for_lines, "image1_lines")
# save_image(image2_for_lines, "image2_lines")
image_for_lines = np.concatenate((image1_for_lines, image2_for_lines), axis=1)
save_image(image_for_lines, "res08")


