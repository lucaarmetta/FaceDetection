import cv2
import matplotlib.pyplot as plt
import os

def detect_and_display_faces(image_path):
    # construct the path to the Haar cascade XML file
    cascade_path = '/Users/larmetta001/Downloads/haarcascade_frontalface_default.xml'
    # check if the cascade file exists
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
    # read the input image from the provided path
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")
    # convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # load the Haar Cascade classifier for face detection from the provided path
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    # detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(
        gray_image,  # input image (grayscale)
        scaleFactor=1.1,  # parameter for scaling the image in each step
        minNeighbors=5,  # how many neighbors each rectangle should have to retain it
        minSize=(40, 40)  # minimum size of the face to be detected
    )
    # draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    # convert the image from BGR to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # display the image with the detected faces highlighted
    plt.figure(figsize=(20, 10))  # set the figure size
    plt.imshow(img_rgb)  # show the image in RGB
    plt.axis('off')  # hide the axis for a cleaner display
    plt.show()  # display the image

# example usage
detect_and_display_faces('/Users/larmetta001/Downloads/face.jpg')