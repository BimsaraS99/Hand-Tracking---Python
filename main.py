import HandTracking
import cv2

cap = cv2.VideoCapture(0)
detector = HandTracking.Hand_Detector()

while True:
    success, img = cap.read()
    image_now = detector.find_hand(img=img, draw=True)
    lm_list, hand_detect = detector.find_hand_position(img=img, draw=True, draw_id=10)

    if hand_detect:
        print(lm_list[4])

    cv2.imshow("Image", image_now)
    cv2.waitKey(1)
