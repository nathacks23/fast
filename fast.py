import cv2
import dlib
import numpy as np

# # Function to convert dlib shape to NumPy array
# def shape_to_np(shape, dtype="int"):
#     coords = np.zeros((shape.num_parts, 2), dtype=dtype)
#     for i in range(0, shape.num_parts):
#         coords[i] = (shape.part(i).x, shape.part(i).y)
#     return coords

def get_mouth_landmarks(shape):    
    left_mouth_indices = [48, 49, 50, 58, 59, 60, 61, 67]
    right_mouth_indices = [52, 63, 65, 56, 53, 64, 55, 54]
    
    right_mouth_landmarks = [shape.part(index) for index in right_mouth_indices]
    left_mouth_landmarks = [shape.part(index) for index in left_mouth_indices]

    # return mouth_landmarks
    return right_mouth_landmarks, left_mouth_landmarks

def is_droopy_face(mouth_landmarks):
    # Split the mouth landmarks into left and right halves
    left_mouth = mouth_landmarks[:len(mouth_landmarks)//2]
    right_mouth = mouth_landmarks[len(mouth_landmarks)//2:]

    # Calculate the average y-coordinate for each half
    left_avg_y = sum(point.y for point in left_mouth) / len(left_mouth)
    right_avg_y = sum(point.y for point in right_mouth) / len(right_mouth)

    # Set a threshold for droopiness (adjust as needed)
    droopiness_threshold = 5  # You may need to adjust this value based on your observations
    print(abs(left_avg_y - right_avg_y))

    # Compare the average y-coordinates to determine droopiness
    droopy_face = abs(left_avg_y - right_avg_y) > droopiness_threshold

    return droopy_face

# Set the threshold value
threshold = 5  # You can adjust this value based on your requirements

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the image
image = cv2.imread('no_droop2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# # Assume there is only one face in the image
# for face in faces:
#     shape = predictor(gray, face)
#     landmarks = shape_to_np(shape)

#     # Define relevant landmarks for facial symmetry
#     left_eye_landmarks = landmarks[36:42]
#     right_eye_landmarks = landmarks[42:48]
    
#     # left_mouth_landmarks = landmarks[]
    
#     # print(landmarks)

#     # Calculate the distance between relevant landmarks
#     left_eye_distance = np.mean(np.linalg.norm(left_eye_landmarks - np.roll(left_eye_landmarks, 1, axis=0), axis=1))
#     right_eye_distance = np.mean(np.linalg.norm(right_eye_landmarks - np.roll(right_eye_landmarks, 1, axis=0), axis=1))
    
#     print(left_eye_distance)
#     print(right_eye_distance)

#     # Compare the distances
#     if abs(left_eye_distance - right_eye_distance) > threshold:
#         print("Possible face drooping detected!")
#     else:
#         print("nope")

for face in faces:
    # Predict facial landmarks for each face
    shape = predictor(gray, face)

    # Get mouth landmarks
    # mouth_landmarks = get_mouth_landmarks(shape)
    right_mouth_landmarks, left_mouth_landmarks = get_mouth_landmarks(shape)
    

    # Check for droopy face
    # droopy_face = is_droopy_face(mouth_landmarks)
    # print(droopy_face)

    for point in right_mouth_landmarks:
        x, y = point.x, point.y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
    for point in left_mouth_landmarks:
        x, y = point.x, point.y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Display the result
    # cv2.putText(image, f"Droopy Face: {droopy_face}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Droopy Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Display the image with landmarks
# for (x, y) in landmarks:
#     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# cv2.imshow('Facial Landmarks', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
