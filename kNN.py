import cv2 as cv
import numpy as np




def findCharacterKNN(array):
    # Load ảnh dữ liệu
    srcImg = cv.imread("Data.png", 0)
    cells = [np.hsplit(row, 50) for row in np.vsplit(srcImg, 30)]
    x = np.array(cells)
    # Tạo dữ liệu train
    train = x[:,:].reshape(-1,1500).astype(np.float32)
    # Gán nhãn cho dữ liệu train
    labels = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9', 'A', 'B',
    'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T',
    'U', 'V', 'X', 'Y', 'Z']
    k = np.arange(30)
    trainLabels = np.repeat(k, 50)[:, np.newaxis]
    # Khởi tạo KNN
    kNN = cv.ml.KNearest_create()
    kNN.train(train, cv.ml.ROW_SAMPLE,trainLabels)
    result = "" # Kết quả thu được
    for i in (range(len(array))):
        kq = kNN.findNearest(array[i], 5)[1]  # Trả về nhãn
        result += labels[int(kq)]   
    return result