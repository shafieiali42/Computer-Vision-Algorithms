import math

import cv2 as cv
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class_names = listdir("Data/Train")
train_images = []
train_labels = []
test_images = []
test_labels = []

for class_name in class_names:
    for file in listdir(f"Data/Train/{class_name}/"):
        image = cv.imread(f"Data/Train/{class_name}/{file}")
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        train_images.append(image)
        train_labels.append(class_name)

    for file in listdir(f"Data/Test/{class_name}/"):
        image = cv.imread(f"Data/Test/{class_name}/{file}")
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        test_images.append(image)
        test_labels.append(class_name)

maximum=0
descriptors=[]
for image in train_images:
    sift=cv.SIFT_create(150)
    kp,des=sift.detectAndCompute(image,None)
    des=des.tolist()
    for i in des:
        descriptors.append(i)


def create_histograms(images,km_model,number_of_clusters):
    histograms=[]
    sift=cv.SIFT.create()
    counter=0
    for image in images:
        print(counter)
        counter=counter+1
        # hist=np.zeros((number_of_clusters),dtype="int32")
        kp,des=sift.detectAndCompute(image,None)
        des=des.astype("float64")
        labels = km_model.predict(des)
        labels=labels.tolist()
        hist=[]
        for i in range(number_of_clusters):
            cnt=labels.count(i)
            hist.append(cnt)
        hist = np.array(hist)
        hist=hist/np.sum(hist)
        histograms.append(hist)

    histograms=np.array(histograms)
    return histograms



number_of_clusters=100
kmeans_model=KMeans(n_clusters=number_of_clusters,max_iter=200)
descriptors=np.array(descriptors)
kmeans_model.fit(descriptors)

train_histograms=create_histograms(train_images,kmeans_model,number_of_clusters)
test_histograms=create_histograms(test_images,kmeans_model,number_of_clusters)


def train(x_train_dataset,train_label,k,distance_function):
    knnModel = KNeighborsClassifier(n_neighbors=k, weights="uniform", p=distance_function)
    knnModel.fit(x_train_dataset,train_label)
    return knnModel

train_labels=np.array(train_labels)
test_labels=np.array(test_labels)
knn_model=train(train_histograms,train_labels,10,1)
test_score = knn_model.score(test_histograms,test_labels)
train_score = knn_model.score(train_histograms,train_labels)
print(f"Train accuracy is: {train_score}")
print(f"Test accuracy is: {test_score}")
