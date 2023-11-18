import cv2
import dlib
import numpy as np

# Function to convert dlib shape to NumPy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Set the threshold value
threshold = 5  # You can adjust this value based on your requirements

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the image
image = cv2.imread('download.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Assume there is only one face in the image
for face in faces:
    shape = predictor(gray, face)
    landmarks = shape_to_np(shape)

    # Define relevant landmarks for facial symmetry
    left_eye_landmarks = landmarks[36:42]
    right_eye_landmarks = landmarks[42:48]

    # Calculate the distance between relevant landmarks
    left_eye_distance = np.mean(np.linalg.norm(left_eye_landmarks - np.roll(left_eye_landmarks, 1, axis=0), axis=1))
    right_eye_distance = np.mean(np.linalg.norm(right_eye_landmarks - np.roll(right_eye_landmarks, 1, axis=0), axis=1))

    # Compare the distances
    if abs(left_eye_distance - right_eye_distance) > threshold:
        print("Possible face drooping detected!")

# Display the image with landmarks
for (x, y) in landmarks:
    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

cv2.imshow('Facial Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
