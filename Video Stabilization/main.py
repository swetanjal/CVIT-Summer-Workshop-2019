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
    camera_theta = []
    frame_nos = []
    camera_x.append(0)
    camera_y.append(0)
    camera_theta.append(0)
    frame_nos.append(0)
    while success:
        success, image = vidObj.read()
        if success == False:
            break
        tmp = f(prev_image, image)
        print("Frame: ", count)
        camera_x.append(camera_x[count - 1] + tmp[0][2])
        camera_y.append(camera_y[count - 1] + tmp[1][2])
        camera_theta.append(camera_theta[count - 1] + math.atan2(tmp[1, 0], tmp[0,0]))
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
    with open('theta.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([camera_theta])
    plt.plot(camera_x, frame_nos)
    plt.xlabel('Motion in x') 
    plt.title('Motion in x over frames')
    plt.show()
    plt.plot(camera_y, frame_nos)
    plt.xlabel('Motion in y') 
    plt.title('Motion in y over frames')
    plt.show()
    plt.plot(camera_theta, frame_nos)
    plt.xlabel('Angle') 
    plt.title('Angle vs frames')
    plt.show()

def stabilize(path):
    vidObj = cv2.VideoCapture(path)
    success, image = vidObj.read()
    y_threshold = int(0.1 * len(image))
    x_threshold = int(0.1 * len(image[0]))
    #print(x_threshold)
    #print(y_threshold)
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
    f = open('theta.csv', 'r')
    csvreader = csv.reader(f)
    theta = []
    for rows in csvreader:
        for i in rows:
            theta.append(float(i))
    plt.plot(X)
    plt.show()
    plt.plot(Y)
    plt.show()
    plt.plot(theta)
    plt.show()
    fx = cp.Variable(len(X))
    fy = cp.Variable(len(Y))
    ft = cp.Variable(len(theta))
    theta_threshold = 0.02
    constraints = [cp.abs(fx - X) <= x_threshold,
                    cp.abs(fy - Y) <= y_threshold,
                    cp.abs(ft - theta) <= theta_threshold]
    # D
    D = 0
    for i in range(len(X)):
        D = D + ((fx[i] - X[i])**2) + ((fy[i] - Y[i])**2) + ((ft[i] - theta[i])**2)
    D = D / 2
    # L11
    L11 = 0
    for i in range(len(X) - 1):
        L11 = L11 + cp.abs((fx[i + 1] - fx[i])) + cp.abs((fy[i + 1] - fy[i])) + cp.abs((ft[i + 1] - ft[i]))
    # L12
    L12 = 0
    for i in range(len(X) - 2):
        L12 = L12 + cp.abs((fx[i + 2] - 2 * fx[i + 1] + fx[i])) + cp.abs((fy[i + 2] - 2 * fy[i + 1] + fy[i])) + cp.abs((ft[i + 2] - 2 * ft[i + 1] + ft[i]))
    # L13
    L13 = 0
    for i in range(len(X) - 3):
        L13 = L13 + cp.abs((fx[i + 3] - 3 * fx[i + 2] + 3 * fx[i + 1] - fx[i])) + cp.abs((fy[i + 3] - 3 * fy[i + 2] + 3 * fy[i + 1] - fy[i])) + cp.abs((ft[i + 3] - 3 * ft[i + 2] + 3 * ft[i + 1] - ft[i]))
    lambda1 = 10000
    lambda2 = 1000
    lambda3 = 100000
    obj = cp.Minimize(D + lambda1 * L11 + lambda2 * L12 + lambda3 * L13)
    sol = cp.Problem(obj, constraints)
    sol.solve()
    plt.plot(X)
    plt.plot(fx.value)
    plt.show()
    plt.plot(Y)
    plt.plot(fy.value)
    plt.show()
    plt.plot(theta)
    plt.plot(ft.value)
    plt.show()
    with open('stablised_x.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([fx.value])
    with open('stabilised_y.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([fy.value])
    with open('stabilised_theta.csv', 'w') as FILE:
        writer = csv.writer(FILE)
        writer.writerows([ft.value])
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

    f = open('stabilised_theta.csv', 'r')
    csvreader = csv.reader(f)
    smooth_theta = []
    for rows in csvreader:
        for i in rows:
            smooth_theta.append(float(i))

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
    f = open('theta.csv', 'r')
    csvreader = csv.reader(f)
    theta = []
    for rows in csvreader:
        for i in rows:
            theta.append(float(i))
    plt.plot(X)
    plt.plot(smooth_X)
    plt.show()
    plt.plot(Y)
    plt.plot(smooth_Y)
    plt.show()
    plt.plot(theta)
    plt.plot(smooth_theta)
    plt.show()
    vidObj = cv2.VideoCapture(path)
    success, image = vidObj.read()
    currY = int(len(image) / 10)
    currX = int(len(image[0]) / 10)
    W = int(0.8 * len(image[0]))
    H = int(0.8 * len(image))
    cnt = 1
    while success:
        success, image = vidObj.read()
        ##############################
        (h, w) = image.shape[:2]
        #(h, w) = (0, 0)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -(smooth_theta[cnt] - theta[cnt]) * 180 / math.pi, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        ##############################
        x_shift = int(X[cnt] - smooth_X[cnt])
        y_shift = int(Y[cnt] - smooth_Y[cnt])
        cv2.rectangle(image,(currX + x_shift, currY + y_shift),(x_shift + W + currX, y_shift + H + currY), (0,0,255), 3)
        cv2.imshow('test', image[currY + y_shift : y_shift + H + currY, currX + x_shift : x_shift + W + currX, :])
        #cv2.imshow('test', image)
        cv2.waitKey(20)
        if success == False:
            break
        if cnt == 1000:
            break
        cnt = cnt + 1
    cv2.destroyAllWindows()
#FrameCapture('./movie.mp4')
#stabilize('./movie.mp4')
generate_video('./movie.mp4')