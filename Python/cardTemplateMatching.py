import cv2
import argparse
import numpy as np

#接收命令行参数
ap=argparse.ArgumentParser(description="示例程序")
ap.add_argument("-i","--input",required=True,help="待处理图像")
ap.add_argument("-t","--template",required=True,help="检测使用模板")
args=vars(ap.parse_args())

print("检测模板："+args["template"])
print("待处理图像："+args["input"])

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#识别数字图像排序函数
def sort_contours(cnts,method="left-to-right"): # 默认从左至右
    if cnts.count == 0:
        return []
    boundingBoxes = [cv2.boundingRect(c)for c in cnts]
    (cnts,boundingBoxes) = zip(*sorted(zip(cnts,boundingBoxes),
                                       key=lambda b: b[1][0],reverse=False))
    #sorted需要传入一个函数参数key，所以必须使用lambda表达式
    return (cnts,boundingBoxes)


img=cv2.imread(args["template"])
#cv_show("template",img)

#进行处理，得到数字模板
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值处理
binary_img=cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)[1]
#cv_show("binary_img",binary_img)

contours,hierarchy=cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
_img=img.copy()
res=cv2.drawContours(_img,contours,-1,(0,255,0),1)
cv_show('res',res)

#对读取的参数进行排序
contours=sort_contours(contours,"left-to-right")[0]
digits={}

#遍历每一个轮廓
for (i,c) in enumerate(contours):
    x,y,w,h=cv2.boundingRect(c)
    roi=_img[y:y+h,x:x+w]
    roi=cv2.resize(roi,(80,120))
    #cv_show("roi",roi)
    digits[i]=roi

#开始处理图像
img=cv2.imread(args["input"])
#img=cv2.resize(img,(800,600)) #调节适当大小便于后续开运算处理
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv_show("gray",gray)
#binary_img=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)[1]
#cv_show("binary_img",binary_img)

#形态学操作使同一信息的图像合并在一起
#这里创建3*7的算子，因为信息一般横向，需要更大宽度，测试后3*7最为合适
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,12))
sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
Kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
#只进行开操作会导致较多的边框出现
#opening=cv2.morphologyEx(binary_img,cv2.MORPH_OPEN,kernel)
#cv_show("opening",opening)

erosion=cv2.erode(gray,rectKernel,iterations=1)
cv_show("erosion",erosion)

#tophat适合操作纹理较多的图像
#tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
#cur_img=gray-tophat
#cv_show('cur_img',cur_img)

#滤波操作
_,binary_img=cv2.threshold(erosion,127,255,cv2.THRESH_BINARY)
cv_show('dst',binary_img)

gradX=cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,Kernel)
gradX=np.absolute(gradX)
cv_show('gradX_1',gradX)

#这段代码并未起到很好的图像处理作用，对于归一化的使用仍需学习
#归一化操作
#(minVal,maxVal)=(np.min(gradX),np.max(gradX))
#gradX=255*(gradX-minVal)/(maxVal-minVal)
#gradX=gradX.astype("uint8")
#cv_show("gradX",gradX)

#绘制轮廓
contours=cv2.findContours(gradX,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
_img=img.copy()
res=cv2.drawContours(_img,contours,-1,(255,0,127),2)
cv_show('res',res)

#调试之后使用二值处理可以大量减少无关轮廓

#筛选合适长宽比的轮廓
locs=[]
for (i,c) in enumerate(contours):
    x,y,w,h=cv2.boundingRect(c)
    print(i,x,y,w,h)
    ar=w/float(h)
    if ar>5 and ar<7:
        locs.append((x,y,w,h))

for (x,y,w,h) in locs:
    print(x,y,w,h)

output=[]
for(i,(gX,gY,gW,gH)) in enumerate(locs):

    groupOutpput=[]

    group=img[gY:gY+gH,gX:gX+gW]
    #group=cv2.resize(group,(240,40))
    cv_show('group',group)

    gray=cv2.cvtColor(group,cv2.COLOR_BGR2GRAY)
    binary_group=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)[1]
    cv_show('group_to_see',binary_group)
    group_digitCnts,hierarchy=cv2.findContours(binary_group,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    _group=group.copy()
    _group=cv2.drawContours(group,group_digitCnts,-1,(255,0,0),1)
    cv_show('_group',_group)
    digitCnts=sort_contours(group_digitCnts,method="left-to-right")[0]

    #检测模板数值
    for c in digitCnts:
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(80,120))
        #cv_show('roi',roi)

        scores=[]

        for(digit,digitROI) in digits.items():
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF) # 模板匹配
            _,score,_,_=cv2.minMaxLoc(result)
            print(score)
            scores.append(score)
        groupOutpput.append(str(np.argmax(scores)))

    cv2.rectangle(img,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(255,0,0),1)
    cv2.putText(img,"".join(groupOutpput),(gX,gY-15),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)

    output.extend(groupOutpput)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()