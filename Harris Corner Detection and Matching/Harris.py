import cv2 as cv
import numpy as np


image1 = cv.imread("im01.jpg")
image2 = cv.imread("im02.jpg")
image1 = image1.astype("float64")
image2 = image2.astype("float64")


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


def calc_gradient(image, gradientImageName):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobel_x = np.max(sobel_x, axis=2)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    sobel_y = np.max(sobel_y, axis=2)
    gradient_image = sobel_x ** 2 + sobel_y ** 2
    gradient_image = np.sqrt(gradient_image)
    save_image(gradient_image, gradientImageName)
    return sobel_x, sobel_y, gradient_image


def save_image(image, name):
    image = my_clip(image, True)
    image = image.astype("uint8")
    cv.imwrite(f"{name}.jpg", image)


def non_maximum_suppression(image, min_distance):
    points = []
    while np.max(image) > 0:
        point = np.where(image == np.max(image))
        point = (point[0][0], point[1][0])
        points.append(point)
        x1_indx = max(0, point[0] - min_distance)
        x2_indx = min(image.shape[0], point[0] + min_distance)
        y1_indx = max(0, point[1] - min_distance)
        y2_indx = min(image.shape[1], point[1] + min_distance)
        image[x1_indx:x2_indx + 1, y1_indx:y2_indx + 1] = 0
    return np.array(points)


def draw_interest_points(image, points, resultName):
    counter = 0
    for point in points:
        cv.drawMarker(image, (point[1], point[0]), color=(0, 255, 255), thickness=10)
        counter = counter + 1

    save_image(image, resultName)


def calc_tensor_structure(image, kernelSize, kernelSigma, k, threshold, gradientImageName, scoreImageName,
                          scoreThreshImage, resultName):
    i_x, i_y, _ = calc_gradient(image, gradientImageName)
    i_x2 = i_x * i_x
    i_y2 = i_y * i_y
    i_xy = i_x * i_y
    s_x2 = cv.GaussianBlur(i_x2, (kernelSize, kernelSize), kernelSigma)
    s_y2 = cv.GaussianBlur(i_y2, (kernelSize, kernelSize), kernelSigma)
    s_xy = cv.GaussianBlur(i_xy, (kernelSize, kernelSize), kernelSigma)
    det = s_x2 * s_y2 - s_xy * s_xy
    trace = s_x2 + s_y2
    R = det - k * trace * trace
    rShow = my_clip(np.copy(R), cast=True)
    save_image(rShow, f"{scoreImageName}")
    R = np.where(R > threshold, R, 0)
    rShow = my_clip(np.copy(R), cast=True)
    save_image(rShow, f"{scoreThreshImage}")
    interest_points = non_maximum_suppression(R, 100)
    draw_interest_points(np.copy(image), interest_points, resultName)
    return interest_points


def create_feature_vectors(image, interest_points, neighbour_size):
    feature_vectors = []
    for point in interest_points:
        x1 = max(0, point[0] - neighbour_size)
        x2 = min(image.shape[0], point[0] + neighbour_size)
        y1 = max(0, point[1] - neighbour_size)
        y2 = min(image.shape[1], point[1] + neighbour_size)
        patch = image[x1:x2 + 1, y1:y2 + 1]
        vector = patch.reshape(-1)
        feature_vectors.append(vector)

    return feature_vectors


def calc_distance(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    return distance


def create_distance_matrix(feature_vectors1, feature_vectors2):
    matrix = np.zeros((len(feature_vectors1), len(feature_vectors2)))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i][j] = calc_distance(feature_vectors1[i], feature_vectors2[j])
    return matrix


def calc_two_nearest_vectors(distance_matrix):
    first = np.zeros((distance_matrix.shape[0], 2), dtype="int32")
    index = np.argsort(distance_matrix)[:, 0]
    first[:, 0] = index
    index = np.argsort(distance_matrix)[:, 1]
    first[:, 1] = index
    #######
    second = np.zeros((distance_matrix.shape[1], 2), "int32")
    index = np.argsort(distance_matrix, axis=0)[0, :]
    second[:, 0] = index
    index = np.argsort(distance_matrix, axis=0)[1, :]
    second[:, 1] = index
    return first, second


def calculate_valid_point(first_matrix, second_matrix, distance_matrix, ratio_threshold):
    for i in range(first_matrix.shape[0]):
        d1 = distance_matrix[i, first_matrix[i, 0]]
        d2 = distance_matrix[i, first_matrix[i, 1]]
        if float(d1 / d2) < ratio_threshold:
            image1_array[i] = first_matrix[i, 0]

    for i in range(second_matrix.shape[0]):
        d1 = distance_matrix[second_matrix[i, 0], i]
        d2 = distance_matrix[second_matrix[i, 1], i]
        if float(d1 / d2) < ratio_threshold:
            image2_array[i] = second_matrix[i, 0]

    for i in range(image1_array.shape[0]):
        a = image1_array[i]
        for j in range(image1_array.shape[0]):
            if a == image1_array[j] and i != j:
                image1_array[i] = -1
                image1_array[j] = -1

    for i in range(image2_array.shape[0]):
        a = image2_array[i]
        for j in range(image2_array.shape[0]):
            if a == image2_array[j] and i != j:
                image2_array[i] = -1
                image2_array[j] = -1

    list_of_pairs = []
    for i in range(image1_array.shape[0]):
        if image1_array[i] != -1 and image2_array[image1_array[i]] == i:
            list_of_pairs.append((i, image1_array[i]))

    return list_of_pairs


def draw_corresponding_points(image1, image2, list_of_pairs):
    for pair in list_of_pairs:
        cv.drawMarker(image1, (interest_points1[pair[0]][1], interest_points1[pair[0]][0]), color=(255, 0, 0),thickness=10)
        cv.drawMarker(image2, (interest_points2[pair[1]][1], interest_points2[pair[1]][0]), color=(255, 0, 0),thickness=10)

    save_image(image1, "res09_corres.jpg")
    save_image(image2, "res10_corres.jpg")


interest_points1 = calc_tensor_structure(image1, 19, 5, 0.05, 10000000, "res01_grad", "res03_score", "res05_thresh",
                                         "res07_harris")

interest_points2 = calc_tensor_structure(image2, 19, 5, 0.05, 10000000, "res02_grad", "res04_score", "res06_thresh",
                                         "res08_harris")

feature_vector1 = create_feature_vectors(image1, interest_points1, 20)
feature_vector2 = create_feature_vectors(image2, interest_points2, 20)
distance_matrix = create_distance_matrix(feature_vector1, feature_vector2)
first_matrix, second_matrix = calc_two_nearest_vectors(distance_matrix)

image1_array = np.ones(interest_points1.shape[0], dtype="int32")
image1_array = image1_array * -1

image2_array = np.ones(interest_points2.shape[0], dtype="int32")
image2_array = image2_array * -1

list_of_pairs = calculate_valid_point(first_matrix, second_matrix, distance_matrix, 0.9)
draw_corresponding_points(np.copy(image1), np.copy(image2), list_of_pairs)
two_image = np.concatenate([image1, image2], axis=1)
color = [(0, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 0)]
color_counter = 0
for pair in list_of_pairs:
    w = image1.shape[1]
    cv.line(two_image, (interest_points1[pair[0]][1], interest_points1[pair[0]][0]),
            (interest_points2[pair[1]][1] + w, interest_points2[pair[1]][0]), color=color[color_counter % 7],thickness=5)

    color_counter = color_counter + 1

save_image(two_image, "res11.jpg")
