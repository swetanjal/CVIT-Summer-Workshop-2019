import cv2
import numpy as np
import math
import os
import csv
import cvxpy as cp
import matplotlib.pyplot as plt

def f(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k = 2)
    pts1 = []
    pts2 = []
    good = []
    c = 0
    # Apply ratio test
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key = lambda x: x.distance)
    for i in range(len(good)):
        pts1.append([kp1[good[i].trainIdx].pt[0], kp1[good[i].trainIdx].pt[1]])
        pts2.append([kp2[good[i].queryIdx].pt[0], kp2[good[i].queryIdx].pt[1]])
    pts1 = np.asarray(pts1, dtype = np.float32)
    pts2 = np.asarray(pts2, dtype = np.float32)
    M, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    return M

def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success, prev_image = vidObj.read()
    count += 1
    camera_x = []
    camera_y = []
    frame_nos = []
    camera_x.append(0)
    camera_y.append(0)
    frame_nos.append(0)
    while success:
        success, image = vidObj.read()
        if success == False:
            break
        tmp = f(prev_image, image)
        print("Frame: ", count)
        camera_x.append(camera_x[count - 1] + tmp[0][2])
        camera_y.append(camera_y[count - 1] + tmp[1][2])
        prev_image = image
        count += 1
        frame_nos.append(count)
        if count == 1001:
            break
    with open('x.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([camera_x])
    with open('y.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([camera_y])
    plt.plot(camera_x, frame_nos)
    plt.xlabel('Motion in x') 
    plt.title('Motion in x over frames')
    plt.show()
    plt.plot(camera_y, frame_nos)
    plt.xlabel('Motion in y') 
    plt.title('Motion in y over frames')
    plt.show()

def stabilize():
    f = open('x.csv', 'r')
    csvreader = csv.reader(f)
    X = []
    for rows in csvreader:
        for i in rows:
            X.append(float(i))
    f = open('y.csv', 'r')
    csvreader = csv.reader(f)
    Y = []
    for rows in csvreader:
        for i in rows:
            Y.append(float(i))    
    plt.plot(X)
    plt.show()
    plt.plot(Y)
    plt.show()

    fx = cp.Variable(len(X))
    fy = cp.Variable(len(Y))
    constraints = []
    # D
    D = 0
    for i in range(len(X)):
        D = D + ((fx[i] - X[i])**2) + ((fy[i] - Y[i])**2)
    D = D / 2
    # L11
    L11 = 0
    for i in range(len(X) - 1):
        L11 = L11 + cp.abs((fx[i + 1] - fx[i])) + cp.abs((fy[i + 1] - fy[i]))
    # L12
    L12 = 0
    for i in range(len(X) - 2):
        L12 = L12 + cp.abs((fx[i + 2] - 2 * fx[i + 1] + fx[i])) + cp.abs((fy[i + 2] - 2 * fy[i + 1] + fy[i]))
    # L13
    L13 = 0
    for i in range(len(X) - 3):
        L13 = L13 + cp.abs((fx[i + 3] - 3 * fx[i + 2] + 3 * fx[i + 1] - fx[i])) + cp.abs((fy[i + 3] - 3 * fy[i + 2] + 3 * fy[i + 1] - fy[i]))
    lambda1 = 1000
    lambda2 = 100
    lambda3 = 10000
    obj = cp.Minimize(D + lambda1 * L11 + lambda2 * L12 + lambda3 * L13)
    sol = cp.Problem(obj, constraints)
    sol.solve()
    plt.plot(X)
    plt.plot(fx.value)
    plt.show()
    plt.plot(Y)
    plt.plot(fy.value)
    plt.show()
    with open('stablised_x.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([fx.value])
    with open('stabilised_y.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([fy.value])
def generate_video(path):
    f = open('stablised_x.csv', 'r')
    csvreader = csv.reader(f)
    smooth_X = []
    for rows in csvreader:
        for i in rows:
            smooth_X.append(float(i))
    f = open('stabilised_y.csv', 'r')
    csvreader = csv.reader(f)
    smooth_Y = []
    for rows in csvreader:
        for i in rows:
            smooth_Y.append(float(i))

    f = open('x.csv', 'r')
    csvreader = csv.reader(f)
    X = []
    for rows in csvreader:
        for i in rows:
            X.append(float(i))
    f = open('y.csv', 'r')
    csvreader = csv.reader(f)
    Y = []
    for rows in csvreader:
        for i in rows:
            Y.append(float(i))    
    plt.plot(X)
    plt.plot(smooth_X)
    plt.show()
    plt.plot(Y)
    plt.plot(smooth_Y)
    plt.show()
    vidObj = cv2.VideoCapture(path)
    success, image = vidObj.read()
    currY = int(len(image) / 4)
    currX = int(len(image[0]) / 4)
    W = 2 * currX
    H = 2 * currY
    cnt = 1
    while success:
        success, image = vidObj.read()
        x_shift = int(X[cnt] - smooth_X[cnt])
        y_shift = int(Y[cnt] - smooth_Y[cnt])
        cv2.rectangle(image,(currX + x_shift, currY + y_shift),(x_shift + W + currX, y_shift + H + currY), (0,0,255), 3)
        cv2.imshow('test', image[currY + y_shift : y_shift + H + currY, currX + x_shift : x_shift + W + currX, :])
        cv2.waitKey(20)
        if success == False:
            break
        if cnt == 1000:
            break
        cnt = cnt + 1
    cv2.destroyAllWindows()
#FrameCapture('./movie.mp4')
#stabilize()
generate_video('./movie.mp4')