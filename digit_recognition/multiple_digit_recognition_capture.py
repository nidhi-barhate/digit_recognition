import cv2
import numpy as np
from keras.models import load_model

model = load_model('models/mnistCNN.h5')

def get_numbers(y_pred):
    for number, per in enumerate(y_pred[0]):
        if per != 0:
            final_number = str(int(number))
            per = round((per * 100), 2)
            return final_number, per


video = cv2.VideoCapture(0)
if(video.isOpened()):
    while True:
        check, img = video.read()
        cv2.imshow("Frame", img)

        #Display purpose
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("Frame thersh",thresh)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imshow("Frame dilation", thresh)

        edged = cv2.Canny(dilation, 50, 250)
        cv2.imshow("Frame edged", thresh)

        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key & 0xFF == ord('c'):
            cv2.imwrite('output/capture.jpg',img)
            capture_img = cv2.imread('output/capture.jpg')

            img2 = capture_img.copy()
            img_gray = cv2.cvtColor(capture_img, cv2.COLOR_RGB2GRAY)
            img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
            ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)

            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(thresh, kernel, iterations=1)

            edged = cv2.Canny(dilation, 50, 250)

            _, contours, hierachy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            num_str = ''
            per = ''

            num_list = []
            if len(contours) > 0:
                for c in contours:
                    if cv2.contourArea(c) > 2500:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 3)

                        new_img = thresh[y:y + h, x:x + w]
                        new_img2 = cv2.resize(new_img, (28, 28))
                        im2arr = np.array(new_img2)
                        im2arr = im2arr.reshape(1, 28, 28, 1)
                        y_pred = model.predict(im2arr)

                        num, per = get_numbers(y_pred)
                        num_list.append(str(int(num)))
                        num_str = '[' + str(str(int(num))) + ']'
                        cv2.putText(img2, num_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

            str1 = ' '.join(num_list)
            if (str1 != ''):
                y_p = str('Predicted Value is ' + str(str1))
                print(y_p)
                cv2.putText(img2, y_p, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Capture Frame", img2)
            cv2.imshow("Contours Frame", thresh)

video.release()
cv2.destroyAllWindows()