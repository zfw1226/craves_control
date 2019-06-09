import cv2
import threading

class camCapture:
    def __init__(self, URL):

        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(URL)
        self.status, self.Frame = self.capture.read()
    def start(self):
        print('usb cam started!')
        t = threading.Thread(target=self.queryframe, args=())
        t.setDaemon(True)
        t.start()

    def stop(self):
        self.isstop = True
        print('usb cam stopped!')

    def getframe(self):
        return self.Frame

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
            # cv2.imshow('frame', self.Frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        self.capture.release()


def video_capture(imgs):
    cap = cv2.VideoCapture(0)
    while (True):
        # capture frame-by-frame
        ret, frame = cap.read()
        if imgs.qsize() >= 2:
            imgs.get()
        imgs.put(frame)
        # imgs.data = torch.ones((640, 480, 3))
        # display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # when everything done , release the capture
    cap.release()