import cv2
import dlib
import numpy as np


def get_mouth_landmarks(shape):    
    left_mouth_indices = [48, 49, 50, 58, 59, 60, 61, 67]
    right_mouth_indices = [52, 63, 65, 56, 53, 64, 55, 54]
    
    right_mouth_landmarks = [shape.part(index) for index in right_mouth_indices]
    left_mouth_landmarks = [shape.part(index) for index in left_mouth_indices]

    # return mouth_landmarks
    return right_mouth_landmarks, left_mouth_landmarks


def is_droopy_face(right_mouth, left_mouth):
    # TODO: need to untilt image
    # Calculate the average y-coordinate for each half
    left_avg_y = sum(point.y for point in left_mouth) / len(left_mouth)
    right_avg_y = sum(point.y for point in right_mouth) / len(right_mouth)

    # Set a threshold for droopiness (adjust as needed)
    droopiness_threshold = 5  # You may need to adjust this value based on your observations
    print(abs(left_avg_y - right_avg_y))

    # Compare the average y-coordinates to determine droopiness
    droopy_face = abs(left_avg_y - right_avg_y) > droopiness_threshold

    return droopy_face


def main():
    # Load the pre-trained facial landmark predictor
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Load the image
    image = cv2.imread('no_droop3.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    for face in faces:
        # Predict facial landmarks for each face
        shape = predictor(gray, face)

        # Get mouth landmarks
        right_mouth_landmarks, left_mouth_landmarks = get_mouth_landmarks(shape)
        
        # Check for droopy face
        droopy_face = is_droopy_face(right_mouth_landmarks, left_mouth_landmarks)
        print(droopy_face)

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
    
    
if __name__ == "__main__":
    main()