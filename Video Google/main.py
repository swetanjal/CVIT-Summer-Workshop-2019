import cv2
import numpy as np
import math
import os
import csv
from sklearn.cluster import KMeans
THRESHOLD = 25
NO_OF_SIFT_FEATURES_FROM_EACH_FRAME = 500
NO_OF_CLUSTERS = 500
NO_OF_FRAMES = 944
def FrameCapture(path):
    """ Extract frames """
    vidObj = cv2.VideoCapture(path)
    count = 0
    success, prev_image = vidObj.read()
    cv2.imwrite("frame%d.jpg" % count, prev_image)
    count += 1
    while success:
        success, image = vidObj.read()
        if success == False:
            break
        diff = np.array(image) - np.array(prev_image)
        diff = diff * diff
        diff = sum(sum(sum(diff)))
        diff = math.sqrt(diff)
        if diff > THRESHOLD:
            cv2.imwrite("frame%d.jpg" % count, image)
            count += 1
            prev_image = image

def detectFeatures():
    """ Detect SIFT features from each frame """
    with open('features.csv', 'w') as f:
        f.write("")
    f.close()
    with open('word_freq.csv', 'w') as f:
        f.write("")
    f.close()
    count = 0
    while os.path.isfile("frame%d.jpg" % count):
        img = cv2.imread("frame%d.jpg" % count, 1)
        sift= cv2.xfeatures2d.SIFT_create(NO_OF_SIFT_FEATURES_FROM_EACH_FRAME)
        kp, des = sift.detectAndCompute(img, None)
        if len(kp) == 0:
            count += 1
            with open('word_freq.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerows([[0]])
            f.close()
            continue
        with open('features.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows(des)
        f.close()
        with open('word_freq.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerows([[des.shape[0]]])
        f.close()
        print("Done with frame %d" % count)
        count += 1

def kmeans():
    """Run K-Means on the 128 dimensional visual words"""
    f = open('features.csv', 'r')
    csvreader = csv.reader(f)
    cnt = 0
    features = []
    for rows in csvreader:
        features.append(rows)
        cnt += 1
    features = np.array(features)
    print(features.shape)
    K_MEANS = KMeans(n_clusters = NO_OF_CLUSTERS, random_state = 0).fit(features)
    with open('means.csv', 'w') as f:
        f.write("")
    f.close()
    with open('means.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(K_MEANS.cluster_centers_)
    f.close()
def create_index_table():
    f = open('means.csv', 'r')
    csvreader = csv.reader(f)
    cnt = 0
    means = []
    for rows in csvreader:
        means.append(rows)
        cnt += 1
    means = np.array(means)
    K_MEANS = KMeans(n_clusters = NO_OF_CLUSTERS, random_state = 0)
    K_MEANS.cluster_centers_ = means
    print(K_MEANS.cluster_centers_.shape)
    table = []
    for i in range(NO_OF_CLUSTERS):
        table.append({})
    count = 0
    while os.path.isfile("frame%d.jpg" % count):
        img = cv2.imread("frame%d.jpg" % count, 1)
        sift= cv2.xfeatures2d.SIFT_create(NO_OF_SIFT_FEATURES_FROM_EACH_FRAME)
        kp, des = sift.detectAndCompute(img, None)
        if len(kp) == 0:
            count += 1
            continue
        res = K_MEANS.predict(des)
        print("Done with frame %d" % count)
        for el in res:
            if count in table[el]:
                table[el][count] += 1
            else:
                table[el][count] = 1
        count += 1
    with open('inverted_index_table.csv', 'w') as f:
        f.write("")
    f.close()
    main = []
    freq = []
    for i in range(NO_OF_CLUSTERS):
        t = []
        f = []
        for el in table[i]:
            t.append(el)
            f.append(table[i][el])
        main.append(t)    
        freq.append(f)
    with open('inverted_index_table.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(np.array(main))
    f.close()

    with open('frequency_table.csv', 'w') as f:
        f.write("")
    f.close()
    with open('frequency_table.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(np.array(freq))
    f.close()
def query():
    img = cv2.imread('query.jpg')
    sift= cv2.xfeatures2d.SIFT_create(NO_OF_SIFT_FEATURES_FROM_EACH_FRAME)
    kp, des = sift.detectAndCompute(img, None)
    f = open('means.csv', 'r')
    csvreader = csv.reader(f)
    cnt = 0
    means = []
    for rows in csvreader:
        means.append(rows)
        cnt += 1
    means = np.array(means)
    K_MEANS = KMeans(n_clusters = NO_OF_CLUSTERS, random_state = 0)
    K_MEANS.cluster_centers_ = means

    res = K_MEANS.predict(des)
    no_of_words = len(res)
    bag = []
    for i in range(NO_OF_CLUSTERS):
        bag.append(0)
    for el in res:
        bag[el] += 1
    f = open('inverted_index_table.csv', 'r')
    csvreader = csv.reader(f)
    tmp = []
    for rows in csvreader:
        tmp.append(rows)
    f = open('frequency_table.csv', 'r')
    csvreader = csv.reader(f)
    tmp2 = []
    for rows in csvreader:
        tmp2.append(rows)
    d = []
    for i in range(NO_OF_CLUSTERS):
        d.append({})
    for i in range(NO_OF_CLUSTERS):
        for j in range(len(tmp[i])):
            d[i][int(tmp[i][j])] = int(tmp2[i][j]) 
    for j in range(NO_OF_CLUSTERS):
        bag[j] = (bag[j] / no_of_words) * math.log(NO_OF_FRAMES / len(d[j]))
    f = open('word_freq.csv')
    csvreader = csv.reader(f)
    total_words = []
    for rows in csvreader:
        total_words.append(int(rows[0]))
    result = []
    for i in range(NO_OF_FRAMES):
        doc = []
        if total_words[i] == 0:
            continue
        for j in range(NO_OF_CLUSTERS):
            if i in d[j]:
                doc.append((d[j][i] / total_words[i]) * math.log(NO_OF_FRAMES / len(d[j])))
            else:
                doc.append(0)
        d1 = np.array(doc)
        d2 = np.array(bag)
        angle = np.dot(d1, d2) / (math.sqrt(np.dot(d1, d1)) * math.sqrt(np.dot(d2, d2)))
        angle = math.acos(angle)
        result.append({'frame': i, 'angle': angle})
    result = sorted(result, key = lambda i : i['angle'])
    for i in range(len(result)):
        output_image = cv2.imread('frame%d.jpg' % result[i]['frame'])
        cv2.imshow('Rank %d' % (i + 1), output_image)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        else:
            cv2.destroyAllWindows()
            continue
#FrameCapture('./movie.mkv')
#detectFeatures()
#kmeans()
#create_index_table()
query()