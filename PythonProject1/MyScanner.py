import numpy as np
import cv2
import argparse
import pytesseract
import os

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

#函数
def cv_show(img):
    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def four_points_transform(img,pts):
    #首先对pts点集进行排序
    rect=order_points(pts)
    (tl, tr, br, bl) = rect

    weightA=np.sqrt(((tr[0]-tl[0])**2)+(tr[1]-tl[1])**2)
    weightB=np.sqrt(((br[0]-bl[0])**2)+(br[1]-bl[1])**2)
    Weight=max(int(weightA),int(weightB)) # 这里先强制类型转换或者后前置类型转换对于实验结果没有影响

    heightA=np.sqrt(((tl[0]-bl[0])**2)+(tl[1]-bl[1])**2)
    heightB=np.sqrt(((br[0]-tr[0])**2)+(br[1]-tr[1])**2)
    Height=max(int(heightA),int(heightB))
    dst=np.array([[0,0],[Weight,0],[Weight,Height],[0,Height]],dtype=np.float32)

    #计算转置矩阵
    M=cv2.getPerspectiveTransform(rect,dst)
    warped=cv2.warpPerspective(img,M,(Weight,Height))
    return warped

#将给定的点集按照一定的方式进行排序
def order_points(pts):
    rect=np.zeros((4, 2), dtype="float32")
    #将点按照左上，右上，左下，右下的顺序进行排序
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)] # 左上
    rect[2]=pts[np.argmax(s)] # 右下
    # 这里之后问下老师
    s=np.diff(pts,axis=1) # diff函数似乎是y-x的结果
    rect[1]=pts[np.argmin(s)] # 右上
    rect[3]=pts[np.argmax(s)] # 左下
    return rect


image=cv2.imread(args["image"])

cv_show(image)

#寻找文档部分
#使用了边缘检测而不是图像二值化之后进行边框绘制
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,50,150)

cv_show(edged)

#轮廓检测
cnts,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
image_copy=image.copy()

for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx)==4:
        screenCnt=approx
        img_copy=image.copy()
        cv2.drawContours(img_copy,[screenCnt],-1,(0,255,0),2)
        cv_show(img_copy)
        break

#将图像进行透视变换
warped=four_points_transform(image,screenCnt.reshape(4,2))
cv_show(warped)
cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref=cv2.threshold(warped,127,255,cv2.THRESH_BINARY)[1]

#将结果旋转至正位
rows,cols=ref.shape[:2]
center=(cols/2,rows/2)
angle=90
scale=1
M=cv2.getRotationMatrix2D(center,angle,scale) # 旋转矩阵
rotated_img=cv2.warpAffine(ref,M,(cols,rows))

cv2.imwrite("scan.jpg",rotated_img) # 为什么图像会泛蓝色
cv_show(rotated_img)

#使用tesseract对图像内容进行读取
#真是奇了怪了，不理解
img=cv2.imread('scan.jpg')
#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray=cv2.medianBlur(gray,3)
text=pytesseract.image_to_string(img,lang="eng")
print(text)

print("_________________________________________________________")

preprocess='blur'
image=cv2.imread('scan.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
if preprocess == 'thresh':
    gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
if preprocess=='blur':
    gray=cv2.medianBlur(gray,3)
filename="{}.png".format(os.getpid())
cv2.imwrite(filename,gray)
text=pytesseract.image_to_string(gray,lang='eng')
print(text)
os.remove(filename)
#np.savetxt("text.txt",text)

# 算是完成了