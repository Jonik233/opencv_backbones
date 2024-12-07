import cv2
from cv2 import DescriptorMatcher, Feature2D

class Tracker:
    def __init__(self, detector:Feature2D, matcher:DescriptorMatcher):
        self.matcher = matcher
        self.detector = detector
        
        
    def save_targets(self, gray_img) -> None:
        target_kp, target_des = self.detector.detectAndCompute(gray_img, None)
        self.targets = (target_kp, target_des)
    
    
    def _track_video(self, source_path:str):
        video = cv2.VideoCapture(source_path)
        
        if not video.isOpened():
            print("Error: Cannot open video.")
            exit()
        
        while True:
            res, frame = video.read()
            if not res:
                break
            
            target_kp, target_des = self.targets
            kp, des = self.detector.detectAndCompute(frame, None)
            matches = self.matcher.match(target_des, des) if des is not None else []

            if matches:
                for match in matches:
                    frame_keypoint_indxs = match.trainIdx
                    keypoint_coords = kp[frame_keypoint_indxs].pt
                    x, y = int(keypoint_coords[0]), int(keypoint_coords[1])
                    
                    cv2.circle(frame, (x, y), 10, [0, 255, 0], 3)

            cv2.imshow("Stream", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


    def track(self, source_path:str):
        format = source_path.split(".")[1]
        if format == "mp4":
            self._track_video(source_path)