import streamlit as st
import cv2
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import requests

def func():
    vid = cv2.VideoCapture('http://10.1.75.217:8080/video') #'http://10.1.75.217:8080/video'

    st.title('Ring Size Detection')
    frame_window = st.image([])
    take_picture_button = st.button('Take Picture')

    while True:
        got_frame, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if got_frame:
            frame_window.image(frame)

        if take_picture_button:
            cv2.imwrite('IPWebcam.jpg', frame)
            img = cv2.imread('IPWebcam.jpg', -1)

            rgb_planes = cv2.split(img)

            result_planes = []
            result_norm_planes = []
            for plane in rgb_planes:
                dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
                bg_img = cv2.medianBlur(dilated_img, 21)
                diff_img = 250 - cv2.absdiff(plane, bg_img)
                norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                result_planes.append(diff_img)
                result_norm_planes.append(norm_img)

            result = cv2.merge(result_planes)
            result_norm = cv2.merge(result_norm_planes)

            cv2.imwrite('shadows_out_norm.png', result_norm)


            # Function to show array of images (intermediate results)
            def show_images(images):
                for i, img in enumerate(images):
                    cv2.imshow("image_" + str(i), img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


            img_path = 'shadows_out_norm.png'

            # Read image and preprocess
            img = cv2.imread(img_path)

            scale_percent = 80  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # blur = cv2.GaussianBlur(gray, (9, 9), 0)
            #
            # edged = cv2.Canny(blur, 50, 100)
            # edged = cv2.dilate(edged, None, iterations=1)
            # edged = cv2.erode(edged, None, iterations=1)

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

            # cv2.drawContours(image, cnts, -1, (0,255,0), 3)

            # show_images([image, edged])
            # print(len(cnts))

            # Reference object dimensions
            # Here for reference I have used a 2cm x 2cm square
            print(len(cnts))
            ref_object = cnts[1]
            box = cv2.minAreaRect(ref_object)
            box = cv2.boxPoints(box)
            cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            dist_in_pixel = euclidean(tl, tr)
            dist_in_cm = 22.4
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
                    cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
                    mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
                    mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))

                    # print(wid)

                    # print(ht)
                    wid1 = "{:.4f}mm".format(wid)
                    ht1 = "{:.4f}mm".format(ht)
                    cv2.putText(image, "{:.4f}mm".format(wid),
                                (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 0), 2)
                    cv2.putText(image, "{:.4f}mm".format(ht), (int(mid_pt_verticle[0] + 10),
                                                               int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (250, 255, 0), 2)
            st.title('Detected Object')
            st.image([image])
            st.title('diameter')

            #print(wid1)
            #print(mid_pt_horizontal[0] - 15)
            st.write(wid1)
            st.write(ht1)
            #show_images([image])
            break

func()

continue_button = st.button('Continue')
if continue_button:
    func()
