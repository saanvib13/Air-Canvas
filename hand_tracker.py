import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False):
        self.mode=mode        

        # solutions submodule within MediaPipe, and then further accesses the hands submodule within solutions
        self.mpHands=mp.solutions.hands

        # creates an instance of the Hands class from the mp_hands module, which is part of the MediaPipe library
        # This instance of the Hands class can then be used to perform hand tracking on input data
        self.hands=self.mpHands.Hands() 
        #keeping the default values for the parameters.

        self.mpDraw=mp.solutions.drawing_utils


    def findHands(self, img):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # hands instance only uses rgb color
        self.results=self.hands.process(imgRGB)

        # print("detected hand coordinates: ", results.multi_hand_landmarks)
        # print("which hand: ", results.multi_handedness)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                
                # drawing points on hands
                # mpDraw.draw_landmarks(img,hand)

                # for connections also
                self.mpDraw.draw_landmarks(img,hand, self.mpHands.HAND_CONNECTIONS,landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                            connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1))
        
        return img

    def findPosition(self, img, handNo=0):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for idx, lm in enumerate(myHand.landmark):
                h,w,c=img.shape

                cx,cy=int(lm.x*w), int(lm.y*h)

                lmList.append([idx,cx,cy])

                if(idx==8 or idx==12):
                    cv2.circle(img, (cx,cy), 7, (0,0,0), cv2.FILLED)
        return lmList
    
    def fingers(self,img):
        pos=self.findPosition(img)
        detectUpFingers=[]
        if pos:
            #index
            detectUpFingers.append((pos[8][2]<pos[7][2]) and (pos[7][2]<pos[6][2]))
            #middle
            detectUpFingers.append((pos[12][2]<pos[11][2]) and (pos[11][2]<pos[10][2]))
            # small 
            detectUpFingers.append((pos[20][2]<pos[19][2]) and (pos[19][2]<pos[18][2]))


        return detectUpFingers

def main():

    
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()

    while True: 
        (success,img1)=cap.read()
        img=cv2.flip(img1,1)

        img=detector.findHands(img)
        lmList=detector.findPosition(img)

        if len(lmList)!=0:
            print(lmList,"\n")

        
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img, "fps: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 4, (255,40,255), 2)
        cv2.imshow("Image",img)
        key=cv2.waitKey(1) 
        if key==ord('q'): 
            break

if __name__=="__main__":
    main()