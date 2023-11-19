import cv2
import numpy as np
import keras
                

def main():
    # Load the pre-trained stroke predictor
    model = keras.models.load_model('stroke_recognition.h5')
    
    image = cv2.imread('data/noStroke_data/aug_0_57.jpg')
    # image = cv2.imread(f)
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

    print("Prediction:", prediction_label)        
    
    
if __name__ == "__main__":
    main()