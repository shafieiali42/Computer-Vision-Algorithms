import random

import cv2 as cv
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


image1 = cv.imread("im03.jpg")
image2 = cv.imread("im04.jpg")

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
images_with_interest_points = cv.drawMatches(image1, kp1, image2, kp2, None, None, None, singlePointColor=(0, 255, 0))
save_image(images_with_interest_points, "res13_corners")

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for i in matches:
    if i[0].distance < 0.75 * i[1].distance:
        good_matches.append(i[0])

image_for_corresponding_points = np.copy(images_with_interest_points)
for match in good_matches:
    cv.drawMatches(image1, [kp1[match.queryIdx]], image2, [kp2[match.trainIdx]], None, image_for_corresponding_points
                   , singlePointColor=(255, 0, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

save_image(image_for_corresponding_points, "res14_correspondences")

image_for_match_points = image_for_corresponding_points.copy()
cv.drawMatches(image1, kp1, image2, kp2, good_matches, image_for_match_points
               , matchColor=(255, 0, 0),
               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)

save_image(image_for_match_points, "res15_matches")

random_good_match = random.sample(good_matches, 20)
image_for_20_match_points = image_for_corresponding_points.copy()
cv.drawMatches(image1, kp1, image2, kp2, random_good_match, image_for_20_match_points, matchColor=(255, 0, 0),
               flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

save_image(image_for_20_match_points, "res16")

image1_points = []
image2_points = []
for match in good_matches:
    image1_points.append(kp1[match.queryIdx].pt)
    image2_points.append(kp2[match.trainIdx].pt)

image1_points = np.array(image1_points, dtype="float64")
image2_points = np.array(image2_points, dtype="float64")
homography, mask = cv.findHomography(image2_points, image1_points, method=cv.RANSAC, ransacReprojThreshold=5,
                                     maxIters=1000)
# print(mask)
# print("")
print(homography)
image_for_inliers = cv.drawMatches(image1, None, image2, None, None, None)
cv.drawMatches(image1, kp1, image2, kp2, good_matches, image_for_inliers, matchColor=(255, 0, 0),singlePointColor=(255, 0, 0),
               flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

cv.drawMatches(image1, kp1, image2, kp2, good_matches, image_for_inliers, matchColor=(0, 0, 255),singlePointColor=(0, 0, 255),
               matchesMask=mask,flags=cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
save_image(image_for_inliers, "res17")

homography_inverse = np.linalg.inv(homography)
draw_rect_image = cv.drawMatches(image1, None, image2, None, None, None)
w = image1.shape[1]

p1 = cv.perspectiveTransform(np.float32([[[0, 0]]]), homography_inverse)
p2 = cv.perspectiveTransform(np.float32([[[image1.shape[1], 0]]]), homography_inverse)
p3 = cv.perspectiveTransform(np.float32([[[image1.shape[1], image1.shape[0]]]]), homography_inverse)
p4 = cv.perspectiveTransform(np.float32([[[0, image1.shape[0]]]]), homography_inverse)

p1 = p1.astype("int32")
p2 = p2.astype("int32")
p3 = p3.astype("int32")
p4 = p4.astype("int32")
pts = np.array([p1, p2, p3, p4])
pts = pts.reshape((-1, 1, 2))
image2_with_rect=np.copy(image2)
cv.drawMarker(image2_with_rect, (p1[0][0][0], p1[0][0][1]), color=(0, 0, 255), thickness=8)
cv.drawMarker(image2_with_rect, (p2[0][0][0], p2[0][0][1]), color=(0, 0, 255), thickness=8)
cv.drawMarker(image2_with_rect, (p3[0][0][0], p3[0][0][1]), color=(0, 0, 255), thickness=8)
cv.drawMarker(image2_with_rect, (p4[0][0][0], p4[0][0][1]), color=(0, 0, 255), thickness=8)
cv.polylines(image2_with_rect, [pts], isClosed=True, color=(0, 255, 0))



p1[0][0][0] = p1[0][0][0] + w
p2[0][0][0] = p2[0][0][0] + w
p3[0][0][0] = p3[0][0][0] + w
p4[0][0][0] = p4[0][0][0] + w

p1 = p1.astype("int32")
p2 = p2.astype("int32")
p3 = p3.astype("int32")
p4 = p4.astype("int32")
pts = np.array([p1, p2, p3, p4])
pts = pts.reshape((-1, 1, 2))

cv.drawMarker(draw_rect_image, (p1[0][0][0], p1[0][0][1]), color=(0, 0, 255), thickness=8)
cv.drawMarker(draw_rect_image, (p2[0][0][0], p2[0][0][1]), color=(0, 0, 255), thickness=8)
cv.drawMarker(draw_rect_image, (p3[0][0][0], p3[0][0][1]), color=(0, 0, 255), thickness=8)
cv.drawMarker(draw_rect_image, (p4[0][0][0], p4[0][0][1]), color=(0, 0, 255), thickness=8)
cv.polylines(draw_rect_image, [pts], isClosed=True, color=(0, 255, 0))
save_image(draw_rect_image, "res19")


p1_image2 = cv.perspectiveTransform(np.float32([[[0, 0]]]), homography)
p2_image2 = cv.perspectiveTransform(np.float32([[[image2.shape[1], 0]]]), homography)
p3_image2 = cv.perspectiveTransform(np.float32([[[image2.shape[1], image2.shape[0]]]]), homography)
p4_image2 = cv.perspectiveTransform(np.float32([[[0, image2.shape[0]]]]), homography)

min_x = min(p1_image2[0][0][0], p2_image2[0][0][0], p3_image2[0][0][0], p4_image2[0][0][0])
max_x = max(p1_image2[0][0][0], p2_image2[0][0][0], p3_image2[0][0][0], p4_image2[0][0][0])

min_y = min(p1_image2[0][0][1], p2_image2[0][0][1], p3_image2[0][0][1], p4_image2[0][0][1])
max_y = max(p1_image2[0][0][1], p2_image2[0][0][1], p3_image2[0][0][1], p4_image2[0][0][1])
m_translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
matrix = np.matmul(m_translation, homography)

warped = cv.warpPerspective(image2, matrix, (int(max_x - min_x), int(max_y - min_y)))
save_image(warped, "res20")


warped_with_rect = cv.warpPerspective(image2_with_rect, matrix, (int(max_x - min_x), int(max_y - min_y)))
ratio=image1.shape[0]/image1.shape[1]
image1_resized=np.empty([])
image1_resized=np.copy(image1)
image1_resized=cv.resize(image1,(int(image1.shape[1]*ratio),warped_with_rect.shape[0]),image1_resized)
res21 = np.concatenate([image1_resized, warped_with_rect], axis=1)
save_image(res21,"res21")