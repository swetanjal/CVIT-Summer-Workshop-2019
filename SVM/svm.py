# Import relevant packages
from sklearn.svm import SVC
import numpy as np
from mnist import MNIST

mndata = MNIST('samples')

# Load training dataset
images, labels = mndata.load_training()
l = len(labels)
filtered_labels = []
filtered_images = []
# Filter dataset to contain only 0s and 1s
for i in range(l):
    if labels[i] == 0 or labels[i] == 1:
        filtered_labels.append(labels[i])
        filtered_images.append(images[i])
# Reassign filtered_images, filtered_labels to images, labels respectively.
images = np.array(filtered_images)
labels = np.array(filtered_labels)
# Training phase
clf = SVC(kernel = 'rbf', gamma = 'scale')
clf.fit(images, labels)

# Load testing dataset
images, labels = mndata.load_testing()
l = len(labels)
filtered_labels = []
filtered_images = []
# Filter dataset to contain only 0s and 1s
for i in range(l):
    if labels[i] == 0 or labels[i] == 1:
        filtered_labels.append(labels[i])
        filtered_images.append(images[i])
# Reassign filtered_images, filtered_labels to images, labels respectively.
images = np.array(filtered_images)
labels = np.array(filtered_labels)
# Prediction phase
y_hat = clf.predict(images)
l = len(y_hat)
correct = 0
for i in range(l):
    if y_hat[i] == labels[i]:
        correct = correct + 1
print("Accuracy: ", end = " ")
print((correct / len(y_hat)))