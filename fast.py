import os
import cv2
import dlib
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras

def get_middle_of_face_landmarks(shape):
    # Define the indices for the middle of the face
    middle_indices = [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]

    # Get the landmarks for the middle of the face
    middle_landmarks = [shape.part(index) for index in middle_indices]

    return middle_landmarks

def align_face(image, landmarks):
    # Convert landmarks to a NumPy array
    landmarks = np.array([(point.x, point.y) for point in landmarks], dtype=np.float32)

    # Calculate the angle of rotation to make the line connecting top and bottom landmarks vertical
    angle = np.degrees(np.arctan2(landmarks[-1, 1] - landmarks[0, 1], landmarks[-1, 0] - landmarks[0, 0]))

    # Add 90 degrees to the calculated angle to tilt the image an additional 90 degrees clockwise
    angle -= 90
    
    # Create a rotation matrix
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Use the center of the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1)

    # Apply the rotation to the image
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return aligned_face


def get_mouth_landmarks(shape):    
    left_mouth_indices = [48, 49, 50, 58, 59, 60, 61, 67]
    right_mouth_indices = [52, 63, 65, 56, 53, 64, 55, 54]
    
    right_mouth_landmarks = [shape.part(index) for index in right_mouth_indices]
    left_mouth_landmarks = [shape.part(index) for index in left_mouth_indices]

    # return mouth_landmarks
    return right_mouth_landmarks, left_mouth_landmarks


def is_droopy_face(right_mouth, left_mouth, threshold=1):
    # Calculate the average y-coordinate for each half
    left_avg_y = sum(point.y for point in left_mouth) / len(left_mouth)
    right_avg_y = sum(point.y for point in right_mouth) / len(right_mouth)

    droopiness = abs(left_avg_y - right_avg_y)

    # Compare the average y-coordinates to determine droopiness
    droopy_face = droopiness > threshold

    return droopy_face, droopiness


def classify_image(f):
    # Load the pre-trained stroke predictor
    model = keras.models.load_model('stroke_recognition.h5')
    
    # image = cv2.imread('mini_data/noStroke_data/aug_0_57.jpg')
    image = cv2.imread(f)
    image = cv2.resize(image, (150, 150))

    # Reshape and normalize the image
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0

    # Make the prediction
    prediction = model.predict(image)

    # Determine the class based on the prediction
    if prediction[0][0] >= 0.5:
        prediction_label = "Stroke"
    else:
        prediction_label = "Not Stroke"

    # print("Prediction:", prediction_label)
    
    return prediction_label


                

def main():
    # # Load the pre-trained facial landmark predictor
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    droops = []
    directory = "data/stroke_data/"
    for f in os.listdir(directory)[100:]:
        # print(f)
        prediction = classify_image(directory + f)
        if prediction == "Stroke":
            droops.append(True)
        else:
            droops.append(False)
    print(droops.count(True))  # droopy faces aka stroke
    print(droops.count(False))  # no stroke

    #     # Load the image
    #     # image = cv2.imread('mini_data/stroke_data/aug_0_57.jpg')
    #     # print(f)
    #     image = cv2.imread(directory + f)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # to grayscale

    #     # Detect faces in the image
    #     detector = dlib.get_frontal_face_detector()
    #     faces = detector(gray)
        
    #     for face in faces:
    #         # Predict facial landmarks for each face
    #         shape = predictor(gray, face)
            
    #         # align the face
    #         middle_landmarks = get_middle_of_face_landmarks(shape)
    #         aligned_face = align_face(image, middle_landmarks)
            
    #         shape = predictor(aligned_face, face)

    #         # Get mouth landmarks
    #         right_mouth_landmarks, left_mouth_landmarks = get_mouth_landmarks(shape)
            
    #         # Check for droopy face
    #         droopy_face = is_droopy_face(right_mouth_landmarks, left_mouth_landmarks, threshold=1)
    #         # print("DROOPY:", droopy_face)
    #         droops.append(droopy_face[0])

    #         for point in right_mouth_landmarks:
    #             x, y = point.x, point.y
    #             cv2.circle(aligned_face, (x, y), 2, (0, 255, 0), -1)

    #         for point in left_mouth_landmarks:
    #             x, y = point.x, point.y
    #             cv2.circle(aligned_face, (x, y), 2, (0, 255, 0), -1)
                
                
    #         # # Display the result
    #         # cv2.imshow("Droopy Face Detection", aligned_face)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()
    # print(droops.count(True))  # droopy faces aka stroke
    # print(droops.count(False))  # no stroke
    # # print(droops)
    
        
    
    
if __name__ == "__main__":
    main()