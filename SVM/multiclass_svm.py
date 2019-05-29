# One vs One multiclass SVM
# Import relevant packages
from sklearn.svm import SVC
import numpy as np
from mnist import MNIST

mndata = MNIST('samples')

# Load training dataset
images, labels = mndata.load_training()
l = len(labels)
images = np.array(images)
labels = np.array(labels)
# Training phase
clf = SVC(gamma = 'scale')
clf.fit(images, labels)

# Load testing dataset
images, labels = mndata.load_testing()
l = len(labels)
images = np.array(images)
labels = np.array(labels)
# Prediction phase
y_hat = clf.predict(images)
l = len(y_hat)
correct = 0
for i in range(l):
    if y_hat[i] == labels[i]:
        correct = correct + 1
print("Accuracy: ", end = " ")
print((correct / len(y_hat)))