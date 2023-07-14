import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model for hand gesture recognition
model = load_model('vgg19_model.h5')

# Define the dimensions of the cropped image
crop_size = (224, 224)

# Define the font for displaying the coordinates
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
line_type = 2

# Start the video capture from the default camera
cap = cv2.VideoCapture(0)

# Loop over the frames from the video capture
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the image to extract the skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to remove noise and fill gaps in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the detected contours
    for contour in contours:
        # Fit a bounding rectangle to the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the bounding rectangle is big enough to be a hand
        if w > 100 and h > 100:
            # Crop the hand from the frame
            hand = frame[y:y+h, x:x+w]

            # Resize the hand to the required input size of the model
            hand = cv2.resize(hand, crop_size)

            # Convert the cropped hand to black and white
            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
            _, hand = cv2.threshold(hand, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Get the coordinates of the hand
            x1, y1 = x, y
            x2, y2 = x+w, y+h

            # Display the coordinates on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'({x1}, {y1})', (x1, y1-10), font, font_scale, font_color, line_type)

            # Make a prediction on the cropped hand using the model
            hand = np.expand_dims(hand, axis=0)
            hand = np.expand_dims(hand, axis=-1)
            prediction = model.predict(hand)
            prediction_class = np.argmax(prediction)

            # Display the predicted class on the frame
            if prediction_class == 0:
                class_text = '1'
            elif prediction_class == 1:
                class_text = '2'
            elif prediction_class == 2:
                class_text = '3'
            cv2.putText(frame, class_text, (x1, y1 - 30), font, font_scale, font_color, line_type)

            # Display the resulting frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Exit the program if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
