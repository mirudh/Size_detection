from threading import Thread
import cv2
import time
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import os

#path = 'rings'
#os.chdir(path)
#i=1
wait=0

class WebcamStream:
    # initialization method
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        self.vcap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method to return latest read frame
    def read(self):
        return self.frame

    # method to stop reading frames
    def stop(self):
        self.stopped = True

webcam_stream = WebcamStream(stream_id=0)   # 0 id for main camera,"rtsp://admin1:admin@123@10.1.89.121:554/axis-media/media.amp"
webcam_stream.start()
# processing frames in input stream
num_frames_processed = 0
start = time.time()
while True:
    if webcam_stream.stopped is True:
        break
    else:
        frame = webcam_stream.read()
        scale_percent = 80  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # show_images([blur, edged])

        # Find contours
        # cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Sort contours from left to right as leftmost contour is reference object

        (cnts, _) = contours.sort_contours(cnts)

        # Remove contours which are not large enough
        cnts = [x for x in cnts if cv2.contourArea(x) > 150]

        ref_object = cnts[1]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 29.3
        pixel_per_cm = dist_in_pixel / dist_in_cm

        # Draw remaining contours
        for cnt in cnts:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            wid = euclidean(tl, tr) / pixel_per_cm
            # list = []
            # error = size - wid
            # list.append(error)
            # for i in list:
            #     if -1 < i < 1:
            #         print("error_rate:", i)
            ht = euclidean(tr, br) / pixel_per_cm
            if 50 > wid > 5 and 50 > ht > 5:
                cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)
                mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
                mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))


                cv2.putText(frame, "{:.4f}mm".format(wid),
                            (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 2)
                cv2.putText(frame, "{:.4f}mm".format(ht), (int(mid_pt_verticle[0] + 10),
                                                           int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (250, 255, 0), 2)

                #wait=wait+100

                wid1 = "{:.4f}mm".format(wid)
                ht1 = "{:.4f}mm".format(ht)


    delay = 0.03
    time.sleep(delay)
    num_frames_processed += 1
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key%256 == 32:
        filename = 'frame_1.jpg'
        cv2.imwrite(filename, frame)
        img = cv2.imread(filename)
        cv2.imshow('capture', img)
        print(wid1)
        print(ht1)
    if key == ord('q'):
        break

end = time.time()
webcam_stream.stop()  # stop the webcam stream

# printing time elapsed and fps
elapsed = end - start
fps = num_frames_processed / elapsed