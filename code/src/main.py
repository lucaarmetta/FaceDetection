import cv2
import matplotlib.pyplot as plt
import os
import mimetypes
import imghdr

def detect_faces_in_image(image_path, cascade_path):
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

def detect_faces_in_video(video_path, cascade_path):
    # open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found at {video_path}")
    # load the Haar Cascade classifier for face detection from the provided path
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        faces = face_classifier.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        # draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # display the frame with detected faces
        cv2.imshow('Video - Press Q to exit', frame)
        # exit if 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def detect_and_display_faces(source_path, cascade_path):
    # check if the file exists
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"File not found at {source_path}")
    # check if the cascade file exists
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
        # Use imghdr to check if the file is an image
    if imghdr.what(source_path) is not None:
        # file is an image
        detect_faces_in_image(source_path, cascade_path)
    else:
        # use mimetypes to check if the file is a video based on MIME type
        mime_type, _ = mimetypes.guess_type(source_path)
        if mime_type and mime_type.startswith('video'):
            # file is a video
            detect_faces_in_video(source_path, cascade_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

# example usage
if __name__ == "__main__":
    source = None # specify image or video source path
    cascade = 'haarcascade_frontalface_default.xml' # construct the path to the Haar cascade XML file
    detect_and_display_faces(source, cascade)