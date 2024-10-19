import cv2
import mediapipe as mp
import time

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV for video capture
cap = cv2.VideoCapture(0)

# Function to detect palm gesture
def is_palm_detected(hand_landmarks):
    # Extract coordinates of key landmarks (wrist and fingers)

    if hand_landmarks:
        # Check for open palm gesture by analyzing relative positions of landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]

        # Palm is open if all fingers' tips are above the wrist (i.e., hand is open)
        palm_open = (
            index_tip.y < hand_landmarks.landmark[0].y and
            middle_tip.y < hand_landmarks.landmark[0].y and
            ring_tip.y < hand_landmarks.landmark[0].y and
            pinky_tip.y < hand_landmarks.landmark[0].y
        )

        return palm_open
    return False



# Function to start countdown
def start_countdown():
    print("Palm detected! Starting countdown...")
    time.sleep(3)  # Wait for 3 seconds
    print("Countdown finished! Taking selfie...")

# Function to take selfie
def take_selfie(frame):
    cv2.imwrite("selfie.jpg", frame)
    print("Selfie taken!")

# Main loop for video capture and palm detection
palm_detected = False
countdown_started = False
selfie_taken = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if palm is detected
            palm_detected = is_palm_detected(hand_landmarks)

            # Perform actions based on palm detection
            if palm_detected:
                if not countdown_started:
                    countdown_started = True
                    start_countdown()
                elif countdown_started and not selfie_taken:
                    selfie_taken = True
                    take_selfie(frame)
            else:
                countdown_started = False
                selfie_taken = False

    cv2.imshow('Image', frame)

    # Exit on key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()