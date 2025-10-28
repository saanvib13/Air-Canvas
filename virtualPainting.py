import cv2
import time
import os
import numpy as np
import random
import hand_tracker as htm
import math

dir=r"D:\air canvas\images"

#  reesizing the shapes images only once -----------------------------------------------------------------------------

# for f in os.listdir(dir):
#     img_path=os.path.join(dir,f)
#     shapes=cv2.imread(img_path)
#     # resized_shape=cv2.resize(shapes,(100,75))

#     # cv2.imwrite(img_path,resized_shape)

#     # height,width,_=resized_shape.shape

#     height,width,_=shapes.shape
#     print(height,width)

# ---------------------------------------------------------------------------------------------------------------------

# class for drawing color boxes ---------------------------------------------------------------------------------------
tool='doodle'
class ColorRect():
    def __init__(self,x,y,w,h,color,text='', alpha=0.5):
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.color=color
        self.text=text
        self.alpha=alpha 

    
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1):
        alpha=self.alpha
        bg_rec=img[self.y:self.y+self.h, self.x:self.x+self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color

        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        
        # Putting the image back to its position
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - tetx_size[0][0]/2), int(self.y + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text,text_pos , fontFace, fontScale,text_color, thickness)

    def isOver(self,x,y):
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False
    
# ---------------------------------------------------------------------------------------------------------------------

# class for drawing shapes icons --------------------------------------------------------------------------------------

class Shape_Rect():
    def __init__(self,img_path,tool,alpha1=0.5):
        self.tool=tool
        self.alpha1=alpha1
        self.img_path=img_path

    def draw_Rect(self,img,i):
        shapes=cv2.imread(self.img_path)
        height,width,_=shapes.shape

        alpha1=self.alpha1
        bg_rec=img[120+i:height+120+i,0:width]
        res = cv2.addWeighted(bg_rec, alpha1, shapes, 1-alpha1, 1.0)
        img[120+i:height+120+i,0:width]=res

    def isOver(self,x,y):
        if (100 > x > 0) and (420> y >120):
            return True
        return False

# ---------------------------------------------------------------------------------------------------------------------

# function for creating mask for hands --------------------------------------------------------------------------------

def generate_mask(frame,lmlist):
    mask = np.zeros_like(frame[:,:,0])
    hand_region_pts=[]
    for landmark in lmlist:
        x, y = landmark[1], landmark[2]
        hand_region_pts.append((x, y))
    cv2.fillPoly(mask,[np.array(hand_region_pts)], (255))

    return mask

# ---------------------------------------------------------------------------------------------------------------------

# function for creating mask for color menu ---------------------------------------------------------------------------

def color_mask(frame,colors,clear):
    mask = np.zeros_like(frame[:,:,0])
    color_region=[]
    color_region.append((colors[0].x,colors[0].y))
    
    y_2=clear.y
    xw_2=clear.x+clear.w
    yh_2=clear.y+clear.h
    color_region.append((xw_2,y_2))
    color_region.append((xw_2,yh_2))

    color_region.append((colors[0].x,colors[0].y+colors[0].h))

    cv2.fillPoly(mask,[np.array(color_region)], (255))

    return mask

# ---------------------------------------------------------------------------------------------------------------------

# function for creating mask for select tool and done menu ------------------------------------------------------------

def menu_mask(frame,select,done):
    mask = np.zeros_like(frame[:,:,0])
    menu_region=[]
    menu_region.append((done.x,done.y))
    
    y_2=select.y
    xw_2=select.x+select.w
    yh_2=select.y+select.h
    menu_region.append((xw_2,y_2))
    menu_region.append((xw_2,yh_2))

    menu_region.append((done.x,done.y+done.h))

    cv2.fillPoly(mask,[np.array(menu_region)], (255))

    return mask

# ---------------------------------------------------------------------------------------------------------------------

# function for blurring the image ------------------------------------------------------------------------------------- 
  
def blur_image(frame,hand_mask,col_mask,men_mask):
    combined_mask=cv2.bitwise_or(hand_mask,col_mask)
    combined_mask = cv2.bitwise_or(combined_mask, men_mask)

    bg_mask = cv2.bitwise_not(combined_mask)

    blur_cap=cv2.GaussianBlur(frame,(15,15),0)

    result = cv2.bitwise_and(frame, frame, mask=hand_mask)
    result+= cv2.bitwise_and(frame, frame, mask=col_mask)
    result+= cv2.bitwise_and(frame, frame, mask=men_mask)

    result += cv2.bitwise_and(blur_cap, blur_cap, mask=bg_mask)

    return result

# ---------------------------------------------------------------------------------------------------------------------

# function for calculating radius -------------------------------------------------------------------------------------
def calc_distance(px,py,fpx,fpy):
    dist=math.sqrt(((fpx-px)**2 ) +((fpy-py)**2))

    return int(dist) 


# ---------------------------------------------------------------------------------------------------------------------


detector=htm.handDetector()
cap=cv2.VideoCapture(0)

cap.set(3,720)
cap.set(4,720)

# print(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

canvas = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)

paintingWindow = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))+255

select=ColorRect(420,350,150,50,(0,0,0),'select tool')
done=ColorRect(300,350,100,50,(0,0,0),"Done")

###################### color, eraser and clear buttons ########################

colors = []

#red
colors.append(ColorRect(100,10,50,50,(0,0,255)))
# yellow
colors.append(ColorRect(160,10,50,50,(0,255,255)))
# blue
colors.append(ColorRect(220,10,50,50,(255,0,0)))
# green
colors.append(ColorRect(280,10,50,50,(0,255,0)))

# eraser
eraser=ColorRect(400,10,50,50,(0,0,0),'eraser')
# clear
clear=ColorRect(460,10,50,50,(100,100,100),'Clear')

# #################################### brush sizes ####################################

brush_size=5

brush_size_btns = []

brush_size_btns.append(ColorRect(550,60,50,50,(20,20,20), '5'))

brush_size_btns.append(ColorRect(550,120,50,50,(20,20,20), '10'))

brush_size_btns.append(ColorRect(550,180,50,50,(20,20,20), '15'))

brush_size_btns.append(ColorRect(550,240,50,50,(20,20,20),'20'))

brush_size_btns.append(ColorRect(550,300,50,50,(20,20,20),'25'))

################################### shapes ####################################

all_shapes=[]
j=0
tools=['circle','doodle','rectangle','triangle']
for f in os.listdir(dir):
    img_path=os.path.join(dir,f)     
    all_shapes.append(Shape_Rect(img_path,tools[j]))
    j+=1          



pTime=0
cTime=0

hideColors=True
px,py=0,0
fpx,fpy=0,0
f=0

color=(255,0,0)

while True:
    (success,img1)=cap.read()
    img=cv2.flip(img1,1)

    img=detector.findHands(img)
    lmList=detector.findPosition(img)
    upFingers=detector.fingers(img)

    select.drawRect(img)
    done.drawRect(img)

    if lmList:
        hand_mask=generate_mask(img,lmList)
        col_mask=color_mask(img,colors,clear)
        men_mask=menu_mask(img,select,done)

        img=blur_image(img, hand_mask,col_mask,men_mask)
        x,y=lmList[8][1], lmList[8][2]
        
        if select.isOver(x,y):
            hideColors=False
            select.alpha=0
        else: select.alpha =0.5

        if done.isOver(x,y):
            hideColors=True
            done.alpha=0
        else: done.alpha=0.5

    
    if not hideColors:
        i=0
        for s in all_shapes:
            s.draw_Rect(img,i)
            i+=75

        for c in colors:
            c.drawRect(img)
            cv2.rectangle(img, (c.x, c.y), (c.x +c.w, c.y+c.h), (255,255,255), 2)

        clear.drawRect(img)
        eraser.drawRect(img)
        cv2.rectangle(img,(clear.x,clear.y), (clear.x+clear.w, clear.y+clear.h),(255,255,255),1)
        cv2.rectangle(img,(eraser.x,eraser.y), (eraser.x+eraser.w, eraser.y+eraser.h),(255,255,255),1)

        for b in  brush_size_btns:
            b.drawRect(img)
            cv2.rectangle(img, (b.x, b.y), (b.x +b.w, b.y+b.h), (255,255,255), 2)


        if lmList:
            x, y = lmList[8][1], lmList[8][2]
            for cb in colors:
                if cb.isOver(x, y):
                    color = cb.color
                    # print(color)
                    cb.alpha = 0
                else:
                    cb.alpha = 0.5

            for bb in brush_size_btns:
                if bb.isOver(x, y):
                    brush_size=int(bb.text)
                    bb.alpha=0
                else: bb.alpha=0.5

            if clear.isOver(x,y):
                clear.alpha=0
                canvas = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
                paintingWindow = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))+255

            else: clear.alpha=0.5

            if eraser.isOver(x,y):
                eraser.alpha=0
                color=(0,0,0)
            else:
                eraser.alpha=0.5

            if 100>x>0:
                if 195>y>120:
                    all_shapes[0].alpha1=0
                    tool='circle'
                else: all_shapes[0].alpha1=0.5
                if 270>y>195:
                    all_shapes[1].alpha1=0
                    tool='doodle'
                else: all_shapes[1].alpha1=0.5
                if 345>y>270:
                    all_shapes[2].alpha1=0
                    tool='rectangle'
                else: all_shapes[2].alpha1=0.5
                if 420>y>345:
                    all_shapes[3].alpha1=0
                    tool='triangle'
                else: all_shapes[3].alpha1=0.5


    if lmList:
        if tool=='doodle':
            if upFingers[0] and not upFingers[1]:
                if px==0 and py==0:
                    px,py=lmList[8][1],lmList[8][2]
                else: 
                
                    if(color==(0,0,0)):
                        cv2.line(canvas,(px,py),(lmList[8][1],lmList[8][2]),color,brush_size*3)
                        cv2.line(paintingWindow,(px,py),(lmList[8][1],lmList[8][2]),(255,255,255),brush_size*3)
                    else:
                    
                        cv2.line(canvas,(px,py),(lmList[8][1],lmList[8][2]),color,brush_size)
                        cv2.line(paintingWindow,(px,py),(lmList[8][1],lmList[8][2]),color,brush_size)
                        # print(color)
    

                px,py=lmList[8][1], lmList[8][2]

            elif upFingers[0] and upFingers[1]:
                px,py=0,0
        
        elif tool=='rectangle':

            if upFingers[0]:
                if(color==(0,0,0)):
                    cv2.line(canvas,(px,py),(lmList[8][1],lmList[8][2]),color,brush_size*3)
                    cv2.line(paintingWindow,(px,py),(lmList[8][1],lmList[8][2]),(255,255,255),brush_size*3)
                elif f==0 and upFingers[1]:
                    px,py=lmList[8][1], lmList[8][2]
                    fpx,fpy=px,py
                elif not upFingers[1]:
                    fpx,fpy=lmList[8][1], lmList[8][2]
                    f=1
                if f==1:
                    if upFingers[1]:
                        cv2.rectangle(canvas,(px,py),(fpx,fpy),color,brush_size)
                        cv2.rectangle(paintingWindow,(px,py),(fpx,fpy),color,brush_size)
                        f=0

            else:
                px,py=0,0
                fpx,fpy=0,0   
                f=0    
            
        elif tool=='circle':
            # print('circle banana')
            if upFingers[0]:
                if(color==(0,0,0)):
                    cv2.line(canvas,(px,py),(lmList[8][1],lmList[8][2]),color,brush_size*3)
                    cv2.line(paintingWindow,(px,py),(lmList[8][1],lmList[8][2]),(255,255,255),brush_size*3)
                if f==0 and upFingers[1]:
                    px,py=lmList[8][1], lmList[8][2]
                    fpx,fpy=px,py
                elif not upFingers[1]:
                    fpx,fpy=lmList[8][1], lmList[8][2]
                    f=1
                if f==1:
                    if upFingers[1]:
                        dist=calc_distance(px,py,fpx,fpy)
                        cv2.circle(canvas,(px,py),dist,color,brush_size)
                        cv2.circle(paintingWindow,(px,py),dist,color,brush_size)


            else:
                px,py=0,0
                fpx,fpy=0,0   
                f=0 

        elif tool=='triangle':
            if upFingers[0]:
                
                if f==0 and upFingers[1]:
                    px,py=lmList[8][1], lmList[8][2]
                    fpx,fpy=px,py
                elif not upFingers[1]:
                    fpx,fpy=lmList[8][1], lmList[8][2]
                    f=1
                if f==1:
                    if upFingers[1]:
                        
                        cv2.line(canvas,(px,py),(fpx,fpy),color,brush_size)
                        cv2.line(paintingWindow,(px,py),(fpx,fpy),color,brush_size)
                        px,py=fpx,fpy
            else:
                px,py=0,0
                fpx,fpy=0,0   
                f=0 

    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Image",img)
    cv2.imshow('paint',paintingWindow)
    key=cv2.waitKey(1) 
    if key==ord('q'): 
        break


# 2ND LOGIC FOR DRAWING A RECTANGLE

    # if upFingers[0] and not upFingers[1]:

            #     if f==0 and upFingers[2]:
            #             px,py=lmList[8][1], lmList[8][2]
            #             fpx,fpy=px,py

            #     elif not upFingers[2]:
            #         fpx,fpy=lmList[8][1], lmList[8][2]
            #         f=1
                
                
            #     if f==1:
            #         if upFingers[2]:
            #             cv2.rectangle(canvas,(px,py),(fpx,fpy),color,brush_size)
            #             cv2.rectangle(paintingWindow,(px,py),(fpx,fpy),color,brush_size)

            #             f=0


            # else: 
            #     px,py=0,0
            #     fpx,fpx=0,0