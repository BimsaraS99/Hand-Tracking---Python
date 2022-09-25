import cv2
import mediapipe as mp


class Hand_Detector:
    def __init__(self, mode=False, max_hand=2, complexity=1, detect_confident=0.5, tracking_confident=0.5):
        self.mode = mode
        self.max_hand = max_hand
        self.complex = complexity
        self.detect_confident = detect_confident
        self.tracking_confident = tracking_confident

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hand, self.complex, self.detect_confident,
                                        self.tracking_confident)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hand(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_hand_position(self, img, hand_no=0, draw=True, draw_id=-1):

        lm_list = []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            select_hand = results.multi_hand_landmarks[hand_no]
            for id_, lm in enumerate(select_hand.landmark):  # handLms.landmark is the list of one hand object
                h, w, c = img.shape  # size of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # hand position X and Y value with respect to image size
                lm_list.append([id_, cx, cy])
                if (draw and id_ == draw_id) or (draw and draw_id == -1):
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lm_list, len(lm_list)






