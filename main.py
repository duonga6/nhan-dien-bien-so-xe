from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox
import cv2 as cv
from PIL import ImageTk, Image
import imgProcess as ip
import kNN as kn



root = Tk()
root.title("Nhận diện biển số xe")
root.geometry("780x400")


# Left Side
leftFrame = Frame(root, width = 210, height = 360)
leftFrame.pack(side= LEFT, fill=Y, padx = 20, pady = 20)

#     Left Title
leftFrameHead = Frame(leftFrame, width=210, height= 40)
leftFrameHead.pack(side = TOP, fill=BOTH)
        # Heading text
headText = Canvas(leftFrameHead, width=210, height=40)
headText.create_text(105, 20, text="KẾT QUẢ", font='Roboto 18 italic bold')
headText.pack()

    # Left Content
leftFrameCont = Frame(leftFrame, width = 210, height = 234)
leftFrameCont.pack(side = TOP, fill = BOTH)

        # Ảnh biển số cắt được
leftContTop = Frame(leftFrameCont, width = 210, height = 100)
leftContTop.pack(side= TOP, fill=BOTH)

            # Phần heading
leftContHeadTOP = Canvas(leftContTop, width=210, height=20)
leftContHeadTOP.create_text(105, 10, text="Ảnh biển số cắt được:", font="Roboto 10 italic bold")
leftContHeadTOP.pack()

            # Phần ảnh
leftContImgTOP = Frame(leftContTop, width = 210, height = 80, highlightbackground="#999", highlightthickness=1)
leftContImgTOP.pack(side = BOTTOM)

        # ký tự tách được

leftContCenter = Frame(leftFrameCont, width=210, height= 80)
leftContCenter.pack(side=TOP)

            # Phần heading
leftContHeadCenter = Canvas(leftContCenter, width=210, height=20)
leftContHeadCenter.create_text(105, 10, text="Ký tự cắt được:", font="Roboto 10 italic bold")
leftContHeadCenter.pack()

            # Phần ảnh
leftContImgCenter = Frame(leftContCenter, width=210, height=60, highlightbackground="#999", highlightthickness=1)
leftContImgCenter.pack(side=BOTTOM)

        # Kết quả sau khi nhận dạng
leftContBOT = Frame(leftFrameCont, width = 210, height = 70)
leftContBOT.pack(side = BOTTOM, fill=BOTH)

            # Phần heading
leftContHeadBOT = Canvas(leftContBOT, width=210, height=20)
leftContHeadBOT.create_text(105, 10, text="Ký tự nhận dạng được:", font="Roboto 10 italic bold")
leftContHeadBOT.pack()

            # Phần ảnh
leftContImgBOT = Frame(leftContBOT, width = 210, height = 50, highlightbackground="#999", highlightthickness=1)
leftContImgBOT.pack(side = BOTTOM)





# Right Size
rightFrame = Frame(root, width = 486, height = 360)
rightFrame.pack(side = RIGHT, padx = 20, pady = 20, fill=BOTH)

    # Ảnh đầu vào
rightFrameTop = Frame(rightFrame, width = 486, height = 304, highlightbackground="#999", highlightthickness=1)
rightFrameTop.pack(side = TOP, fill=BOTH)
    # Các nút chức năng
rightFrameBottom = Frame(rightFrame, width = 486, height = 56)
rightFrameBottom.pack(side = BOTTOM, fill=BOTH)



def loadImg():
    # Clear ảnh

    for label in rightFrameTop.winfo_children():
        label.destroy()
    for label in leftContImgTOP.winfo_children():
        label.destroy()
    for label in leftContImgBOT.winfo_children():
        label.destroy()
    for label in leftContImgCenter.winfo_children():
        label.destroy()

    filetypes = (('JPG', '*.JPG'), ('PNG', '*.PNG'), ('JPEG', '*.JPEG'), ('All files', '*.*'))
    filePath = fd.askopenfilename(title='Chon file anh', initialdir='/', filetypes=filetypes)
    if (filePath == ""):
        messagebox.showinfo("Thông báo", "Bạn chưa chọn ảnh")
        return
    img = Image.open(filePath)
    imgReSize = img.resize((480, 304))
    imgTK = ImageTk.PhotoImage(imgReSize)
    showImg = Label(rightFrameTop, image = imgTK)
    showImg.image = imgTK
    showImg.pack()

    global path
    path = filePath


def processImg(filePath):

    if (filePath == ""):
        messagebox.showinfo("Thông báo", "Bạn chưa chọn ảnh")
        return
    
    # Clear ảnh
    for label in rightFrameTop.winfo_children():
        label.destroy()
    for label in leftContImgTOP.winfo_children():
        label.destroy()
    for label in leftContImgBOT.winfo_children():
        label.destroy()
    for label in leftContImgCenter.winfo_children():
        label.destroy()
    # Xử lý ảnh

    imgOrigin = cv.imread(filePath)
    imgOrigin = cv.resize(imgOrigin, (480, 304), cv.INTER_LINEAR)
    imgLP, imageAfter = ip.imageProcess(imgOrigin, "false")
    cv.cvtColor(imageAfter, cv.COLOR_BGR2RGB, imageAfter)


    imgMainConvert = Image.fromarray(imageAfter)
    imgTKMain = ImageTk.PhotoImage(imgMainConvert)
    showImgMain = Label(rightFrameTop, image = imgTKMain)
    showImgMain.image = imgTKMain
    showImgMain.pack()

    # Show ảnh cắt được

    imgLPConvert = Image.fromarray(imgLP)
    imgLPTK = ImageTk.PhotoImage(imgLPConvert)
    showImgLPTK = Label(leftContImgTOP, image = imgLPTK, width=204, height=76)
    showImgLPTK.image = imgLPTK
    showImgLPTK.pack()

    # Show ky tự cắt được
    characterImagesBinary = ip.findCharacter(imgLP, "false")

    arrImgTK = []
    for i in range(len(characterImagesBinary)):
        convertNp2Pil = Image.fromarray(characterImagesBinary[i])
        convertPil2Tk = ImageTk.PhotoImage(convertNp2Pil)
        arrImgTK.append(convertPil2Tk)

    label = list(range(len(arrImgTK)))
    for i in range(len(arrImgTK)):
        label[i] = Label(leftContImgCenter, image=arrImgTK[i], height=50)
        label[i].image = arrImgTK[i]
        label[i].pack(side=LEFT)

    # Show ket qua
    characterImagesBinaryResized = ip.resizeBinaryImg(characterImagesBinary)
    array = ip.characterImage2Array(characterImagesBinaryResized)
    kq = kn.findCharacterKNN(array)
    characterLP = Canvas(leftContImgBOT, width=204, height=44)
    characterLP.create_text(102, 25, text=kq, font="Roboto 30 bold")
    characterLP.pack()

def showImgProcess(filePath):
    if (filePath == ""):
        messagebox.showinfo("Thông báo", "Bạn chưa chọn ảnh")
        return
    imgOrigin = cv.imread(filePath)
    imgOrigin = cv.resize(imgOrigin, (480, 304), cv.INTER_LINEAR)
    _, _ = ip.imageProcess(imgOrigin, "true")

def showLPProcess(filePath):

    if (filePath == ""):
        messagebox.showinfo("Thông báo", "Bạn chưa chọn ảnh")
        return

    imgOrigin = cv.imread(filePath)
    imgOrigin = cv.resize(imgOrigin, (480, 304), cv.INTER_LINEAR)
    imgLP, _ = ip.imageProcess(imgOrigin, "false")
    _ = ip.findCharacter(imgLP, "true")

# button
openFileButton = ttk.Button(rightFrameBottom, text='Chọn ảnh', command = lambda:loadImg())
openFileButton.pack(anchor = "s", side = RIGHT, ipady = 5, padx=(20,0))


processImgButton = ttk.Button(rightFrameBottom, text='Xử lý', command = lambda:processImg(path))
processImgButton.pack(anchor = "s", side = RIGHT, ipady = 5, padx=(20,0))

showprocessingImg = ttk.Button(rightFrameBottom, text = 'Hiển thị xử lý ảnh', command = lambda:showImgProcess(path))
showprocessingImg.pack(anchor = "s", side = RIGHT, ipady = 5, padx=(20,0))

showprocessingLP = ttk.Button(rightFrameBottom, text = 'Hiện thị tách ký tự', command = lambda:showLPProcess(path))
showprocessingLP.pack(anchor = "s", side = RIGHT, ipady = 5)

exitButton = ttk.Button(leftFrame, text = 'Thoát', command = root.destroy)
exitButton.pack(anchor = "s", side = LEFT, ipady = 5)

root.mainloop()