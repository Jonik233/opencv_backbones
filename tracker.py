import cv2
from cv2 import DescriptorMatcher


class Tracker:
        
    def __init__(self, feature_detector:str, matcher:DescriptorMatcher):
        self.matcher = matcher
        self.feature_detector = feature_detector.lower()

    
    @property
    def feature_detector(self):
        return self._feature_detector
    
    
    @feature_detector.setter
    def feature_detector(self, value):
        valid_detectors = {"fast":"_fast_detector", "sift":"_sift_detector", "orb":"_orb_detector"}
        if value not in valid_detectors:
            raise ValueError(f"Unsupported feature detector: {value}. "
                             f"Choose from {valid_detectors}.")
        
        self._feature_detector = getattr(self, valid_detectors[value])
            
            
    def _fast_detector(self):
        fast = cv2.FastFeatureDetector_create(threshold=20)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        return lambda img: brief.compute(img, fast.detect(img, None))


    def _sift_detector(self):
        sift = cv2.SIFT_create()
        return lambda img: sift.detectAndCompute(img, None)


    def _orb_detector(self):
        orb = cv2.ORB_create()
        return lambda img: orb.detectAndCompute(img, None)

    
    def save_targets(self, gray_img) -> None:
        detector = self.feature_detector()
        self.targets = detector(gray_img)
    
    
    def _track_video(self, source_path:str):
        video = cv2.VideoCapture(source_path)
        detector = self.feature_detector()
        
        if not video.isOpened():
            print("Error: Cannot open video.")
            exit()
        
        while True:
            res, frame = video.read()
            if not res:
                break
            
            target_kp, target_des = self.targets
            kp, des = detector(frame)
            matches = self.matcher.match(target_des, des) if des is not None else []

            if matches:
                for match in matches:
                    frame_keypoint_indxs = match.trainIdx
                    keypoint_coords = kp[frame_keypoint_indxs].pt
                    x, y = int(keypoint_coords[0]), int(keypoint_coords[1])
                    
                    cv2.circle(frame, (x, y), 5, [0, 255, 0], 2)

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