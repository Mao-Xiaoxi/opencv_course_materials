import cv2
import numpy as np
import argparse
from PIL import Image
import pytesseract
import os

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
args=vars(ap.parse_args())

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def order_points(pts):
    #共有四个坐标点
    rect=np.zeros((4,2),dtype="float32")

    #按顺序找到对应的坐标0123分别左上，右上，右下，左下
    #计算左上，右下
    s=pts.sum(axis=1) # 表示将x和y进行求和
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]

    #计算右上和左下
    diff=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(diff)]
    print(np.argmin(diff))
    print(np.argmax(diff))
    rect[3]=pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect=order_points(pts)
    (tl,tr,br,bl)=rect
    widthA=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2)) # 计算两个点之间的欧氏距离
    widthB=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))
    maxWidth=max(int(widthA),int(widthB))

    heightA=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
    heightB=np.sqrt(((tl[0]-bl[0])**2)+((tl[1]-bl[1])**2))
    maxHeight=max(int(heightA),int(heightB))

    #变换后对应的坐标位置
    dst=np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeight-1], # 这里视频中出现错误
        [0,maxHeight-1]
    ],dtype="float32")

    #计算变换矩阵
    M=cv2.getPerspectiveTransform(rect,dst) # 求透视变换矩阵
    warped=cv2.warpPerspective(image,M,(maxWidth,maxHeight)) # 透视变换
    cv_show("warped",warped)
    #返回变换后的结果
    return warped

def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    dim=None
    (h,w)=image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r=height/float(h)
        dim=(int(w*r),height)
    else:
        r=width/float(w)
        dim=(width,int(h*r))
    resized=cv2.resize(image,dim,interpolation=inter)
    return resized

#读取输入
image=cv2.imread(args["image"])
#坐标也会相同变化
ratio=image.shape[0]/500.0
orig=image.copy()

image=resize(orig,height=500)

#预处理
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(5,5),0)
edged=cv2.Canny(gray,75,200) #图像边缘的划分

#显示预处理结果
print("STEP 1:边缘检测")
cv2.imshow("Image",image)
cv2.imshow("Endged",edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

#轮廓检测
cnts,hierarchy=cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)#我草，不知道为什么就整好了，牛逼
cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:5] # 处理之后取前五个元素

#遍历轮廓
for c in cnts:
    #计算轮廓近似
    peri=cv2.arcLength(c,True)
    #C表示输入的点集
    #epsilon表示原始轮廓到近似轮廓的最大距离，他是一个准确参数
    #True表示封闭的
    approx=cv2.approxPolyDP(c,0.02*peri,True) # 进行图像近似

    #4个点的时候取出
    if len(approx)==4:
        screenCnt=approx
        break

#显示结果
print("STEP 2：获取轮廓")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#透视变换
warped=four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
#二值处理
warped=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
ref=cv2.threshold(warped,150,255,cv2.THRESH_BINARY)[1]
cv2.imwrite("scan.jpg",ref)

#将结果旋转至正位
rows,cols=ref.shape[:2]
center=(cols/2,rows/2)
angle=90
scale=1

M=cv2.getRotationMatrix2D(center,angle,scale) # 旋转矩阵
rotated_img=cv2.warpAffine(ref,M,(cols,rows))

resize_img=resize(rotated_img,height=600)

#展示结果
print("STEP 3:变换")
cv2.imshow('Original',orig)
cv2.imshow('Scanned',resize_img)
cv2.waitKey()
cv2.destroyAllWindows()

#多线程可以提高处理多张图片的效率
preprocess ='blur' # thresh

image=cv2.imread('scan.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
if preprocess == 'thresh':
    gray=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
if preprocess=='blur':
    gray=cv2.medianBlur(gray,3)
filename="{}.png".format(os.getpid())
cv2.imwrite(filename,gray)
text=pytesseract.image_to_string(image,lang='eng')
print(text)
os.remove(filename)
#np.savetxt("text.txt",text)