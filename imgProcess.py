import cv2 as cv
from PIL import Image
import numpy as np
import kNN as aa



def numpy2pil(np_array: np.ndarray) -> Image:
    assert isinstance(np_array, np.ndarray)
    assert len(np_array.shape) == 3
    assert np_array.shape[2] == 3

    img = Image.fromarray(np_array, 'RGB')
    return img

def imageProcess(image, flag):

        image_copy = image.copy()
        image_copy2 = image.copy()

        S1 = cv.getStructuringElement(cv.MORPH_RECT, (6,1), (3, 0))
        S2 = cv.getStructuringElement(cv.MORPH_RECT, (3,1), (1,0))
        S3 = cv.getStructuringElement(cv.MORPH_RECT, (5,5), (-1,-1))


        imgNormalize = np.zeros((304, 480, 1), dtype='uint8')           # Ảnh lọc nhiễu
        imgGray = np.zeros((304, 480, 1), dtype='uint8')                # Ảnh xám
        imgMorpho = np.zeros((304, 480, 1), dtype='uint8')              # Ảnh làm rõ các cạnh biên
        imgThreshold = np.zeros((304, 480, 1), dtype='uint8')           # Ảnh nhị phân với ngưỡng trung bình
        imgLPZone = np.zeros((304, 480, 1), 'uint8')                    # Ảnh các vùng có khả năng là biển số (Sau khi lọc nhiễu)
        imgDE = np.zeros((304, 480, 1), dtype='uint8')                  # Ảnh các vùng có khả năng là biển số sau khi
                                                                        # dùng các phép co và nở
        imgContours = np.zeros((304, 480, 3), dtype='uint8')            # Ảnh vẽ các đường bao quanh
        imgContourRect = np.zeros((304, 480, 3), dtype='uint8')         # Ảnh vẽ hình chữ nhật ôm các đường bao quanh
        imgTakeLP = np.zeros((304, 480, 3), dtype='uint8')              # Ảnh vẽ vùng biển số sau khi nhận dạng
        imgLP = np.zeros                                                # Ảnh biển số sau khi tìm được



        # Giảm nhiễu hình ảnh
        # cv.normalize(image_copy, imgNormalize, 0, 255, cv.NORM_MINMAX)

        # Chuyển về ảnh xám
        imgGray = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)

        # Làm nổi bật các cạnh của ảnh
        cv.morphologyEx(imgGray, cv.MORPH_BLACKHAT, S1, imgMorpho)

        cv.normalize(imgMorpho, imgNormalize, 0, 255, cv.NORM_MINMAX)

        # Chuyển ảnh về ảnh nhị phân với ngưỡng trung bình: trung bình các điểm ảnh trắng
        cv.threshold(imgNormalize, 10*cv.mean(imgMorpho)[0] + 10, 255, cv.THRESH_BINARY, imgThreshold)
        # cv.imshow("image_binary", imgThreshold)

        # Chuyển ảnh về ảnh 1 kênh
        imgThreshold.resize(304, 480, 1)



        ## LỌC NHIỄU
        # Sử dụng một hình chữ nhật 16x8 chạy quanh hình ảnh
        # Loại bỏ các vùng có mật độ điểm ảnh trắng thấp
        # Tìm kiếm các vùng có mật độ điểm ảnh cao, có khả năng là biển số
        for i in range(0, 480-32, 4):
                for j in range(0, 304-16, 4): # bước nhảy càng nhỏ khả năng chính xác càng cao

                        # 4 hình chữ nhật 16x8 duyệt quanh hình chữ nhật lớn 32x16

                        white1 = 0
                        rectZone1 = imgThreshold[j:j+8, i:i+16] # Chọn vùng cần xét
                        white1 = cv.countNonZero(rectZone1) # Số điểm ảnh trắng có trong vùng đó

                        white2 = 0
                        rectZone2 = imgThreshold[j:j+8, i+16:i+16+16]
                        white2 = cv.countNonZero(rectZone2)

                        white3 = 0
                        rectZone3 = imgThreshold[j+8:j+8+8, i:i+16]
                        white3 = cv.countNonZero(rectZone3)

                        white4 = 0
                        rectZone4 = imgThreshold[j+8:j+8+8, i+16:i+16+16]
                        white4 = cv.countNonZero(rectZone4)

                        cnt = 0
                        # Nếu số điểm ảnh trắng lớn hơn một ngưỡng nhất định thì vùng đó có khả năng là 1 phần của biển số
                        # Ngưỡng là 15. có thể thay đổi tùy vào nguồn ảnh
                        if (white1 > 15): 
                                cnt+=1
                        if (white2 > 15):
                                cnt+=1
                        if (white3 > 15):
                                cnt+=1
                        if (white4 > 15):
                                cnt+=1
                        
                        # Nếu vùng hình chữ nhật lớn (32x16) có từ 3 vùng có số điểm ảnh trắng cao hơn ngưỡng nhất định
                        # thì đó có khả năng cao là biển số -> lấy vùng đó
                        if (cnt >= 3):
                                imgLPZone[j:j+16, i:i+32] = imgThreshold[j:j+16, i:i+32]


        # cv.imshow("Cac vung co kha nang la bien so",imgLPZone)



        # Sử dụng phép co, phép dãn nở -> các vùng có khả năng là biển số thành một khối


        cv.dilate(imgLPZone, None,imgDE, (-1,-1), 2)
        cv.erode(imgDE, None,imgDE, (-1,-1), 2)
        cv.dilate(imgDE, S2,imgDE, (-1,-1), 9)
        cv.erode(imgDE, S2,imgDE, (-1,-1), 10)
        cv.dilate(imgDE, S3, imgDE)

        # cv.imshow("after_dilate", imgDE)

        # Tìm Contours (đường biên bao quanh cách khối có khả năng là biển số)
        contours, _ = cv.findContours(imgDE, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


        # Chuyển ảnh binary về ảnh RGB để vẽ màu Contours
        imgContours = cv.cvtColor(imgDE, cv.COLOR_GRAY2RGB)
        # Vẽ tất cả contours
        cv.drawContours(imgContours, contours, -1, (0, 0, 255), 1)


        imgContourRect = imgContours.copy()


        for contour in contours:
    
                # Bao quanh các Contours bằng hình chữ nhật, kích thước w,h tọa độ x,y
                (x, y, w, h) = cv.boundingRect(contour)
                # Vẽ hình chữ nhật lên các vùng contours
                cv.rectangle(imgContourRect, (x, y), (x+w, y+h), (255, 0, 0), 1)
        imgTakeLP = imgContourRect.copy()

        for contour in contours:
                (x, y, w, h) = cv.boundingRect(contour)
                # print(w, h)

                # Hình chữ nhật nào có tỉ lệ và kích thước giống với biển số thì đó là BIẾN SỐ
                if (w / h > 2.4 and w / h < 7):   # tỉ lệ dài / rộng ~ 5. Mở rộng phạm vi để tìm kiếm tốt hơn
                        if (w > 50 and w < 120 and h > 10 and h < 40):      # Kích thước dựa trên kích thước biển số ở hình ảnh
                                # Lấy biển số
                                if (len(findCharacter(image[y:y+h, x:x+w], "false")) > 2):
                                        imgLP = image[y:y+h, x:x+w]
                                        # Vẽ Contours vùng biển số
                                        cv.rectangle(imgTakeLP, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                        cv.rectangle(image_copy2, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Tăng kích thước biển số



        if (str(type(imgLP)) == "<class 'builtin_function_or_method'>"):
                imgLP = np.zeros((40, 160, 3), dtype='uint8')
                cv.putText(imgLP, "Khong tim thay", (10,25), cv.FONT_HERSHEY_COMPLEX, .5, (255, 255, 255), 1, cv.LINE_4)
        else:
                imgLP = cv.resize(imgLP, (imgLP.shape[1]*2, imgLP.shape[0]*2))

        if (flag == 'true'):
                cv.imshow("Anh xam", imgGray)
                cv.imshow("Anh lam ro canh", imgMorpho)
                cv.imshow("Anh nhi phan nguong TB", imgThreshold)
                cv.imshow("Anh sau khi loc vung bien so", imgLPZone)
                cv.imshow("Lam ro vung co the la bien so", imgDE)
                cv.imshow("Cac duong vien", imgContours)
                cv.imshow("HCN bao quanh duong vien", imgContourRect)
                cv.imshow("Vung bien so", imgTakeLP)
                cv.imshow("Vung bien so tren anh goc", image_copy2)
                cv.imshow("Bien so cat duoc", imgLP)
                cv.waitKey(0)
                cv.destroyAllWindows()
        return imgLP, image_copy2


def findCharacter(image, flag):
        imgCharacterBinary = []         # Mảng lưu ký tự

        imgBinary = np.zeros((image.shape[0], image.shape[1], 1), dtype='uint8')        # Ảnh biển số sau khi được xử lý
        imgContours = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')      # Vẽ đường contours trên biển số
        imgCharacter = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')     # Vùng ký tự được lấy                      

        # Xử lý biển số
        imgBinary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        imgBinary = cv.adaptiveThreshold(imgBinary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 13, 11)
        S2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        cv.dilate(imgBinary, S2, imgBinary)

        # Vẽ contours
        contours, _ = cv.findContours(imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        imgContours = cv.cvtColor(imgBinary, cv.COLOR_GRAY2BGR)
        cv.drawContours(imgContours, contours, -1, (0, 0, 255), 1)

        class sortImage():
                def __init__(self, img, x):
                        self.img = img
                        self.x = x

        imgCharacter = imgContours.copy()
        arrayBinary = []

        # Vẽ đường bao và tìm ký tự trong biển số
        for contour in contours:
                (x, y, w, h) = cv.boundingRect(contour)
                if (w > 4 and w < 0.20*image.shape[1] and h > 0.35*image.shape[0] and h < 0.85*image.shape[0]):
                        if (h/w > 1.3 and h/w < 5.4):
                                numberZone = imgBinary[y:y+h, x:x+w]
                                white = cv.countNonZero(numberZone)
                                total = w * h
                                if (white / total > 0.2 and white / total < 0.8):
                                        cv.rectangle(imgCharacter, (x, y), (x+w, y+h), (255, 0, 0), 1)
                                        img2ArrayBinary = sortImage(imgBinary[y:y+h, x:x+w], x)
                                        arrayBinary.append(img2ArrayBinary)

        # Sắp xếp lại ký tự thu được
        for i in range(0, len(arrayBinary) - 1):
                for j in range(i + 1, len(arrayBinary)):
                        if (arrayBinary[i].x > arrayBinary[j].x):
                                temp = arrayBinary[i]
                                arrayBinary[i] = arrayBinary[j]
                                arrayBinary[j] = temp


        # Trả về 2 mảng ảnh ký tự: mảng ảnh nhị phân và mảng ảnh rgb
        for i in range(0, len(arrayBinary)):
                imgCharacterBinary.append(arrayBinary[i].img)

        if (flag == "true"):
                cv.imshow("Bien so duoc xu ly", imgBinary)
                cv.imshow("Ve contours", imgContours)
                cv.imshow("Khoanh cac ky tu tren bien so", imgCharacter)
                for i in range(len(imgCharacterBinary)):
                        name = "Character " + str(i)
                        cv.imshow(name, imgCharacterBinary[i])
                cv.waitKey(0)
                cv.destroyAllWindows()
        return imgCharacterBinary

def resizeBinaryImg(array):
        for i in range(len(array)):
                w = array[i].shape[1]
                h = array[i].shape[0]
                ratio = 38/h
                array[i] = cv.resize(array[i], (int(w*ratio), 38))
                temp = np.zeros((50, 30), dtype='uint8')
                temp[6:6+38, int((30-int(w*ratio))/2):int((30-int(w*ratio))/2+int(w*ratio))] = array[i]
                array[i] = temp              

        return array

def characterImage2Array(characterImg):
        arrayCharacter = []
        for i in range(0, len(characterImg)):
                arrayCharacter.append(characterImg[i].reshape(-1, characterImg[i].shape[0] * characterImg[i].shape[1]).astype(np.float32))
        return arrayCharacter
