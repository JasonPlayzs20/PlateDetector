from multiprocessing.resource_tracker import register

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
bg = cv2.createBackgroundSubtractorMOG2()


def register_plate(img):
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    plates = plate_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 30))
    plate_count = 1  # Initialize a counter for saving plate images

    for (x, y, w, h) in plates:
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        x1, y1 = max(0, x - padding_x), max(0, y - padding_y)
        x2, y2 = min(img.shape[1], x + w + padding_x), min(img.shape[0], y + h + padding_y)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plate_crop = img[y1:y2, x1:x2]
        plate_filename = f"plate_{plate_count}.jpg"  # Create a unique filename
        cv2.imwrite(plate_filename, plate_crop)  # Save each detected plate with a unique name
        plate_count += 1  # Increment plate counter

    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # cv2.imshow("iMac Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = bg.apply(gray)

    motionLevel = np.sum(fgmask) / fgmask.size
    if motionLevel > 10:
        print("Motion detected")
        cv2.imwrite("motion.jpg", frame)
        register_plate(frame)  # Call with the color frame
    cv2.imshow("Motion", frame)

cap.release()
cv2.destroyAllWindows()
