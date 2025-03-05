import cv2 as cv
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class_names = listdir("Data/Train")
train_images = []
train_labels = []
test_images = []
test_labels = []

for class_name in class_names:
    for file in listdir(f"Data/Train/{class_name}/"):
        image = cv.imread(f"Data/Train/{class_name}/{file}")
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        train_images.append(image)
        train_labels.append(class_name)

    for file in listdir(f"Data/Test/{class_name}/"):
        image = cv.imread(f"Data/Test/{class_name}/{file}")
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        test_images.append(image)
        test_labels.append(class_name)


def create_dataset(train_data, train_label, test_data, test_label, image_size):
    x_train_dataset = []
    x_test_dataset = []
    for image in train_data:
        resized_image = cv.resize(image, (image_size, image_size))
        x_train_dataset.append(resized_image.flatten())

    for image in test_data:
        resized_image = cv.resize(image, (image_size, image_size))
        x_test_dataset.append(resized_image.flatten())

    x_train_dataset = np.array(x_train_dataset)
    x_test_dataset = np.array(x_test_dataset)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    return x_train_dataset, train_label, x_test_dataset, test_label


def calc_model_accuracy(k, size, distance_function):
    x_train_dataset, train_label, x_test_dataset, test_label = create_dataset(train_images, train_labels, test_images,
                                                                              test_labels, size)

    kFold = KFold(n_splits=5, shuffle=True)
    scores=[]
    for train_part , validation_part in kFold.split(x_train_dataset, train_label):
        knnModel=KNeighborsClassifier(n_neighbors=k,weights="uniform",p=distance_function)
        knnModel.fit(x_train_dataset[train_part],train_label[train_part])
        score=knnModel.score(x_train_dataset[validation_part],train_label[validation_part])
        scores.append(score)

    scores=np.array(scores)
    return np.average(scores)


accuracy_list=[]
size_list=[]
k_list=[]
distance_list=[]
for size in range(4,25,2):
    distance=1
    # for distance in range(1,3):
    accuracy=calc_model_accuracy(1,size,distance)
    accuracy_list.append(accuracy)
    k_list.append(1)
    size_list.append(size)
    distance_list.append(distance)


# plt.plot(size_list,accuracy_list)
# plt.savefig("accuracy_size.jpg")
# plt.clf()
size=size_list[np.argmax(np.array([accuracy_list]))]
print(f"Best size for k=1 is :{size}")


accuracy_list=[]
size_list=[]
k_list=[]
distance_list=[]
for k in range(1,5):
    for distance in range(1,3):
        accuracy=calc_model_accuracy(k,size,distance)
        accuracy_list.append(accuracy)
        k_list.append(k)
        size_list.append(size)
        distance_list.append(distance)

accuracy_list=np.array(accuracy_list)
size_list=np.array(size_list)
k_list=np.array(k_list)
distance_list=np.array(distance_list)
arg_max=np.argmax(accuracy_list)

size=size_list[arg_max]
distance_function=distance_list[arg_max]
k=k_list[arg_max]




x_train_dataset, train_label, x_test_dataset, test_label = create_dataset(train_images, train_labels, test_images,
                                                                          test_labels, size)
knnModel = KNeighborsClassifier(n_neighbors=k, weights="uniform", p=distance_function)
knnModel.fit(x_train_dataset, train_label)
test_score = knnModel.score(x_test_dataset, test_label)
train_score=knnModel.score(x_train_dataset,train_label)
# print(f"Train accuracy : {train_score}")
print(f"Test accuracy is: {test_score}")
print(f"Size: ({size},{size})")
print(f"K: {k}")
print(f"Norm{distance_function}")

print()
print("All together")
print()






accuracy_list=[]
size_list=[]
k_list=[]
distance_list=[]
for size in range(4,25,2):
    for k in range(1,5):
        for distance in range(1,3):
            accuracy=calc_model_accuracy(k,size,distance)
            accuracy_list.append(accuracy)
            k_list.append(k)
            size_list.append(size)
            distance_list.append(distance)


accuracy_list=np.array(accuracy_list)
size_list=np.array(size_list)
k_list=np.array(k_list)
distance_list=np.array(distance_list)
arg_max=np.argmax(accuracy_list)

size=size_list[arg_max]
distance_function=distance_list[arg_max]
k=k_list[arg_max]




x_train_dataset, train_label, x_test_dataset, test_label = create_dataset(train_images, train_labels, test_images,
                                                                          test_labels, size)
knnModel = KNeighborsClassifier(n_neighbors=k, weights="uniform", p=distance_function)
knnModel.fit(x_train_dataset, train_label)
test_score = knnModel.score(x_test_dataset, test_label)
train_score=knnModel.score(x_train_dataset,train_label)
# print(f"Train accuracy : {train_score}")
print(f"Test accuracy is: {test_score}")
print(f"Size: ({size},{size})")
print(f"K: {k}")
print(f"Norm{distance_function}")





