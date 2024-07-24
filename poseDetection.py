import cv2
import time
import mediapipe as mp

class PoseDetection():
    def __init__(self, mode = False, complexity = 1, smooth = True, det_Conf = 0.5, track_Conf = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.det_Conf = det_Conf
        self.track_Conf = track_Conf
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                    model_complexity=self.complexity,
                                    smooth_landmarks=self.smooth,
                                    min_detection_confidence=self.det_Conf,
                                    min_tracking_confidence=self.track_Conf)
        self.mpDraw = mp.solutions.drawing_utils
        self.result = None
        
    def findLandmarks(self,img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(img_rgb)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    
    def getPosition(self, img, draw = True):
        lmlist = []
        if (self.result == None):
            self.findLandmarks(img)
        for id, lm in enumerate(self.result.pose_landmarks.landmark):
            height, width, dims = img.shape
            cx, cy = int(lm.x * width), int(lm.y * height)
            lmlist.append([id, cx, cy])
        if draw:
            cv2.circle(img, (cx, cy), 5, (255,0,0), -1)
        return lmlist

def main():
    cap = cv2.VideoCapture('Videos/5319856-uhd_2160_3840_25fps.mp4')
    pTime = 0
    detector = PoseDetection()
    while (cap.isOpened()):
        success, img = cap.read()
        img = cv2.resize(img, (500,800), interpolation=cv2.INTER_CUBIC)
        if success: 
            result = detector.findLandmarks(img)
            lmlist = detector.getPosition(img, draw=False)
            cv2.circle(result, (lmlist[13][1], lmlist[13][2]), 7, (255,255,0), cv2.FILLED)
            cTime = time.time()
            fps = 1/(cTime - pTime)  
            pTime = cTime  
            cv2.putText(result,"FPS:"+str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
            cv2.imshow("image", result)
            if cv2.waitKey(1) & 0xFF == ord('d'):
                break
        else:
            break
    cap.release() 

if __name__ == "__main__":
    main()