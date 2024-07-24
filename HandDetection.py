import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionConfidence, 
                                        min_tracking_confidence=self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)
        if self.result.multi_hand_landmarks:
            for handLMS in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.result.multi_hand_landmarks:
            myHand_pos = self.result.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand_pos.landmark):
                # print(id,lm)
                height,width,dims = img.shape
                center_x, center_y = int(lm.x*width), int(lm.y*height)
                # print(id,center_x,center_y)
                lmlist.append([id,center_x,center_y])
                
                if draw:
                    cv2.circle(img,(center_x,center_y),10,(255,0,255),-1,cv2.FILLED)
        
        return lmlist

def main():
    
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img,draw=False)
        
        if len(lmlist) != 0:
            print(lmlist[0]) 
        
        cTime = time.time()
        fps = 1/(cTime - pTime)  
        pTime = cTime    
        cv2.putText(img,"FPS:"+str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)     
        cv2.imshow("Live feed",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
if __name__ == "__main__":
    main()