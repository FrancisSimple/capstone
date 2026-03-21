import cv2

class Camera:
    def __init__(self, source="usb"):
        """
        source:
        - "ip"  → phone camera
        - "pi"  → raspberry pi camera
        - "usb" → USB webcam
        """

        self.source = source

        if source == "ip":
            self.url = "http://10.226.232.241:8080/video"
            self.cap = cv2.VideoCapture(self.url)

        elif source == "usb":
            self.cap = cv2.VideoCapture(0)  # ✅ USB webcam

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
        if self.source in ["ip", "usb"]:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

        elif self.source == "pi":
            return self.picam2.capture_array()

    def release(self):
        if self.source in ["ip", "usb"]:
            self.cap.release()

        elif self.source == "pi":
            self.picam2.stop()