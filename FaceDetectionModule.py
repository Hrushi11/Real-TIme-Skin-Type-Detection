import cv2
import time
import mediapipe as mp

class faceDetector():
    def __init__(self, minDetectionCon=0.5, model=0):
        self.minDetectionCon = minDetectionCon
        self.model = model
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img_RGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{str(int(detection.score[0] * 100))}%", org=(bbox[0], bbox[1]-20),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=2, color=(0, 255, 0))

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top left x, y
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    path = "assets/oil-vid.mp4"
    # path = 0
    cap = cv2.VideoCapture(path)
    pTime = 0
    detector = faceDetector(minDetectionCon=0.5)

    while True:
        ret, img = cap.read()
        if not ret:
            cap = cv2.VideoCapture(path)
            ret, img = cap.read()

        img = cv2.resize(img, (800, 480), interpolation=cv2.INTER_AREA)
        img, bboxs = detector.findFaces(img)
        # print(detector.model)

        cTime = time.time()
        if (cTime - pTime) != 0:
            fps = 1 / (cTime - pTime)
            pTime = cTime

        cv2.putText(img, str(int(fps)), org=(15, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2,
                    thickness=3, color=(255, 255, 255))
        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()