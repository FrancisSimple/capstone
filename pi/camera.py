import cv2

class Camera:
    def __init__(self, source="ip"):
        """
        source:
        - "ip"  → phone camera
        - "pi"  → raspberry pi camera
        """

        self.source = source

        if source == "ip":
            # 🔴 CHANGE THIS to your phone IP
            self.url = "http://10.226.232.241:8080/video"
            self.cap = cv2.VideoCapture(self.url)

        elif source == "pi":
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self.picam2.configure(
                self.picam2.create_preview_configuration(
                    main={"size": (640, 480)}
                )
            )
            self.picam2.start()

    def get_frame(self):
        if self.source == "ip":
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

        elif self.source == "pi":
            return self.picam2.capture_array()

    def release(self):
        if self.source == "ip":
            self.cap.release()
        elif self.source == "pi":
            self.picam2.stop()