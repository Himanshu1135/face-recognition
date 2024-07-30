import cv2
import face_recognition as fr
import numpy as np

# Function to load and convert an image
def load_and_convert_image(image_path):
    # Load the image file
    image = cv2.imread(image_path)
    print(f"Loaded image: {image_path}")

    # Check if image is loaded correctly
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Print image type and shape before conversion
    print(f"Image type before conversion: {image.dtype}")
    print(f"Image shape before conversion: {image.shape}")

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Converted image to RGB: {image_path}")

    # Print image type and shape after conversion
    print(f"Image type after conversion: {image_rgb.dtype}")
    print(f"Image shape after conversion: {image_rgb.shape}")
    
    return image_rgb

try:
    # Load and convert the image of CR7
    cr7 = load_and_convert_image("cr7.jpg")

    # Detect face location and encode the face
    loc = fr.face_locations(cr7)[0]
    enco = fr.face_encodings(cr7)[0]

    # Draw a rectangle around the detected face
    cr7 = cv2.rectangle(cr7, (loc[3], loc[0]), (loc[1], loc[2]), (255, 0, 0), 2)

    # Load and convert the test image of CR7
    cr7t = load_and_convert_image("cr7t.jpg")

    # Encode the face in the test image
    encot = fr.face_encodings(cr7t)[0]

    # Compare the encoded faces and compute face distance
    result = fr.compare_faces([enco], encot)
    rtp = fr.face_distance([enco], encot)

    # Annotate the test image with the comparison result
    cr7t = cv2.putText(cr7t, f'{result[0]} {(1-round(rtp[0],2))}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Print the comparison result and face distance
    print(result, (1-rtp)*100)

    # Display the images
    cv2.imshow("img", cr7)
    cv2.imshow("img2", cr7t)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
