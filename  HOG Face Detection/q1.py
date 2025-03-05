import time
import cv2 as cv
import numpy as np
from os import listdir
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
# import pickle

start_time = time.time()


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



np.random.seed(42)


def create_negative_datasets(win_size, block_size, block_stride, cell_size, number_of_bins):
    hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, number_of_bins)
    feature_vectors = []
    for file in listdir("256_ObjectCategories/"):
        for image_name in listdir(f"256_ObjectCategories/{file}/"):
            if image_name[len(image_name) - 3:].lower() != "jpg":
                continue
            image = cv.imread(f"256_ObjectCategories/{file}/{image_name}")
            image = cv.resize(image, win_size)
            feature_vector = hog.compute(image)
            feature_vectors.append(feature_vector.ravel())



    train_x_negative = feature_vectors[:10000]
    train_y_negative = [0 for i in range(len(train_x_negative))]
    validation_x_negative = feature_vectors[10000:11000]
    validation_y_negative = [0 for i in range(len(validation_x_negative))]
    test_x_negative = feature_vectors[11000:12000]
    test_y_negative = [0 for i in range(len(test_x_negative))]

    return train_x_negative, train_y_negative, validation_x_negative, validation_y_negative, test_x_negative, test_y_negative


def create_positive_datasets(win_size, block_size, block_stride, cell_size, number_of_bins):
    hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, number_of_bins)
    feature_vectors = []
    for file in listdir("lfw/"):
        for image_name in listdir(f"lfw/{file}/"):
            image = cv.imread(f"lfw/{file}/{image_name}")
            image = cv.resize(image, win_size)
            feature_vector = hog.compute(image)
            feature_vectors.append(feature_vector.ravel())




    train_x_positive = feature_vectors[:10000]
    train_y_positive = [1 for i in range(len(train_x_positive))]
    validation_x_positive = feature_vectors[10000:11000]
    validation_y_positive = [1 for i in range(len(validation_x_positive))]
    test_x_positive = feature_vectors[11000:12000]
    test_y_positive = [1 for i in range(len(test_x_positive))]
    return train_x_positive, train_y_positive, validation_x_positive, validation_y_positive, test_x_positive, test_y_positive


window_size_list = [(128, 128)]
block_size_list = [(32, 32)]
cell_size_list = [(16, 16)]
number_of_bins_list = [8]
kernel_type_list = ["rbf"]


# for find best parameters

# window_size_list = [(64, 64), (128, 128), (150, 150), (200, 200)]
# block_size_list = [(8, 8), (16, 16), (32, 32)]
# cell_size_list = [(8, 8), (10, 10), (12, 12),(14,14),(16,16)]
# number_of_bins_list = [8, 9, 10]
# kernel_type_list = ["rbf","poly","linear"]



best_window_size = 0
best_block_size = 0
best_cell_size = 0
best_number_of_bins = 0
best_kernel_type = ""
best_validation_score = 0
best_block_stride = 0
best_model = None


def find_best_parameters():
    global best_validation_score
    global best_window_size
    global best_block_size
    global best_cell_size
    global best_number_of_bins
    global best_kernel_type
    global best_block_stride
    global best_model
    for win_size in window_size_list:
        for block_size in block_size_list:
            for cell_size in cell_size_list:
                for number_of_bins in number_of_bins_list:
                    for kernel_type in kernel_type_list:
                        block_stride = cell_size
                        if block_size[0] % cell_size[0] != 0 or block_size[1] % cell_size[1] != 0:
                            continue
                        train_x_positive, train_y_positive, validation_x_positive, validation_y_positive, test_x_positive, test_y_positive = create_positive_datasets(
                            win_size=win_size, block_size=block_size, block_stride=block_stride, cell_size=cell_size,
                            number_of_bins=number_of_bins)

                        train_x_negative, train_y_negative, validation_x_negative, validation_y_negative, test_x_negative, test_y_negative = create_negative_datasets(
                            win_size=win_size, block_size=block_size, block_stride=block_stride, cell_size=cell_size,
                            number_of_bins=number_of_bins)

                        x_train_dataset = train_x_positive + train_x_negative
                        train_label = train_y_positive + train_y_negative

                        x_validation_dataset = validation_x_positive + validation_x_negative
                        validation_label = validation_y_positive + validation_y_negative

                        model = svm.SVC(kernel=kernel_type, C=1, probability=True)
                        model.fit(x_train_dataset, train_label)
                        validation_score = model.score(x_validation_dataset, validation_label)
                        print("model trained!")

                        print(f"window size is : {win_size}")
                        print(f"block size is : {block_size}")
                        print(f"block stride is : {block_stride}")
                        print(f"cell size is : {cell_size}")
                        print(f"number of bins is : {number_of_bins}")
                        print(f"validation accuracy is : {validation_score}")

                        if validation_score > best_validation_score:
                            best_validation_score = validation_score
                            best_cell_size = cell_size
                            best_block_size = block_size
                            best_kernel_type = kernel_type
                            best_window_size = win_size
                            best_number_of_bins = number_of_bins
                            best_block_stride = block_stride
                            best_model = model


def calculate_test_accuracy():
    train_x_positive, train_y_positive, validation_x_positive, validation_y_positive, test_x_positive, test_y_positive = create_positive_datasets(
        win_size=best_window_size, block_size=best_block_size, block_stride=best_block_stride, cell_size=best_cell_size,
        number_of_bins=best_number_of_bins)

    train_x_negative, train_y_negative, validation_x_negative, validation_y_negative, test_x_negative, test_y_negative = create_negative_datasets(
        win_size=best_window_size, block_size=best_block_size, block_stride=best_block_stride, cell_size=best_cell_size,
        number_of_bins=best_number_of_bins)

    x_train_dataset = train_x_positive + train_x_negative
    train_label = train_y_positive + train_y_negative

    x_validation_dataset = validation_x_positive + validation_x_negative
    validation_label = validation_y_positive + validation_y_negative

    x_test_dataset = test_x_positive + test_x_negative
    test_labels = test_y_positive + test_y_negative

    test_score = best_model.score(x_test_dataset, test_labels)
    # train_score=best_model.score(x_train_dataset,train_label)
    validation_score = best_model.score(x_validation_dataset, validation_label)
    # pickle.dump(best_model, open("model_4.sav", 'wb'))
    print(f"best window size is: {best_window_size}")
    print(f"best block size is: {best_block_size}")
    print(f"best block stride is: {best_block_stride}")
    print(f"best cell size is: {best_cell_size}")
    print(f"best number of bins is: {best_number_of_bins}")
    # print(f"Train accuracy is: {train_score}")
    print(f"validation accuracy is: {validation_score}")
    print(f"Test accuracy is: {test_score}")
    return x_test_dataset, test_labels


find_best_parameters()
x_test_dataset, test_labels = calculate_test_accuracy()

score=best_model.decision_function(x_test_dataset)
roc = metrics.roc_curve(test_labels, score)
roc_auc = metrics.roc_auc_score(test_labels, score)
plt.plot(roc[0], roc[1])
plt.title("Roc")
plt.savefig("res1.jpg")

precision_recall = metrics.precision_recall_curve(test_labels, score)
ap = metrics.average_precision_score(test_labels, score)
plt.clf()
plt.plot(precision_recall[0], precision_recall[1])
plt.title(f"precision recall")
plt.savefig("res2.jpg")

print(roc_auc)
print(f"Ap : {ap}")

def find_faces(result_image, resized_image, model, threshold, stride, scale, rectangles):
    hog = cv.HOGDescriptor(best_window_size, best_block_size, best_block_stride, best_cell_size, best_number_of_bins)
    for i in range(0, resized_image.shape[0] - best_window_size[0] + 1, stride):
        for j in range(0, resized_image.shape[1] - best_window_size[1] + 1, stride):
            current_patch = resized_image[i:i + best_window_size[0], j:j + best_window_size[1]]
            current_query = hog.compute(current_patch)
            current_query = current_query.reshape(1, -1)
            response_probability = model.predict_proba(current_query)
            if response_probability[0][1] > threshold:
                up_left = (int(j / scale), int(i / scale))
                down_right = (
                    int(j / scale + best_window_size[1] / scale), int(i / scale + best_window_size[0] / scale))
                rect = [up_left[0], up_left[1], down_right[0], down_right[1]]
                rectangles.append(rect)
    return rectangles


def scale_image(original_image, threshold, stride, model, scales, padding):
    result_image = np.zeros((original_image.shape[0] + padding, original_image.shape[1] + padding, 3), dtype="uint8")
    result_image[int(padding / 2):original_image.shape[0] + int(padding / 2),
    int(padding / 2):original_image.shape[1] + int(padding / 2), :] = np.copy(original_image)
    original_result_image = np.copy(result_image)
    rectangles = []
    for scale in scales:
        print(f"scale: {scale}")
        resized_image = cv.resize(original_result_image, (0, 0), fx=scale, fy=scale)
        rectangles = find_faces(result_image, resized_image, model, threshold, stride, scale, rectangles)

    if len(rectangles) == 0:
        return result_image
    rects = cv.groupRectangles(rectangles, 1, eps=0.1)
    rects = rects[0]
    print(len(rects))

    for i in range(len(rects)):
        result_image = cv.rectangle(result_image, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]),
                                    color=(0,255,0), thickness=4)
    return result_image


image1 = cv.imread("Melli.jpg")
image2 = cv.imread("Persepolis.jpg")
image3 = cv.imread("Esteghlal.jpg")


scales = [1, 0.8, 0.6, 0.5, 0.4, 0.3]
result1 = scale_image(image1, threshold=0.99, stride=5, model=best_model, scales=scales, padding=80)
save_image(result1, "res4")
result2 = scale_image(image2, threshold=0.95, stride=5, model=best_model, scales=scales, padding=150)
save_image(result2, "res5")
result3 = scale_image(image3, threshold=0.99, stride=5, model=best_model, scales=scales, padding=100)
save_image(result3, "res6")

print(time.time() - start_time)
