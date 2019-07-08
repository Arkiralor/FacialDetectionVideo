# Python script to detect face(s) in realtime in live video.

# WARNING! This script may consume a lot of resources depending on the resolution and bit-depth of the video played.

# NOTE: The included Cascade classifier apparently does NOT work well with dar-skinned individuals wearing white clothes
# on a white background.

# Classifier set provided by OpenCV @ https://github.com/opencv/opencv/tree/master/data/haarcascades

# Importing the OpenCV 2.0 library.
import cv2



# Initialising the video variable
# The (0) signifies that the video is not stored in a file and is sourced from any attached video capture device.
vid = cv2.VideoCapture(0)
check, frame = vid.read()
# The read() function returns a tuple, a check (boolean value AFAIK) and the frame which is the current frame.

# Loop counter initialised as '1'.
a = 1

# Creating a classifying object for the individual frames to be compared to. The XML file can be found on Github.
face_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_eye.xml")
glasses_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_eye_tree_eyeglasses.xml")
profile_cascade = cv2.CascadeClassifier("Classifiers/haarcascade_profileface.xml")

# As long as the boolean returned is TRUE i.e, as long as a video playback is detected.
while True:
    # Increment the loop counter by '1'.
    a = a +1

    # Update the check variable and frame from the video buffer.
    check, frame = vid.read()

    # Convert the frame to Black and White and store in a varible
    # (B&W images provide better contrast and have no colour noise).
    gry_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Now, detect the face in the Black and White frame
    # by comparing the B&W frame to the classifying object (training data).
    # The parameters used are the ones used most commonly with this files.
    # Try changing them to see what happens.
    faces = face_cascade.detectMultiScale(gry_img, scaleFactor=1.05, minNeighbors=10)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Blue
        gry_crp = gry_img[y:y + h, x:x + w]
        crp_img = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(gry_crp, scaleFactor=1.05, minNeighbors=5)
        for (x, y, w, h) in eyes:
            cv2.rectangle(crp_img, (x, y), (x + w, y + h), (255, 255, 24), 1)  # Cyan

        glasses = glasses_cascade.detectMultiScale(gry_crp, scaleFactor=1.35, minNeighbors=5)
        for (x, y, w, h) in glasses:
            cv2.rectangle(crp_img, (x, y), (x + w, y + h), (31, 255, 31), 1)  # Green.

    profile_faces = profile_cascade.detectMultiScale(gry_img, scaleFactor=1.25, minNeighbors=5)
    for x, y, w, h in profile_faces:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (17, 255, 255), 1)  # Yellow.
        gry_crp = gry_img[y:y + h, x:x + w]
        crp_img = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(gry_crp, scaleFactor=1.05, minNeighbors=5)
        for (x, y, w, h) in eyes:
            cv2.rectangle(crp_img, (x, y), (x + w, y + h), (255, 255, 24), 1)  # Cyan

        glasses = glasses_cascade.detectMultiScale(gry_crp, scaleFactor=1.35, minNeighbors=5)
        for (x, y, w, h) in glasses:
            cv2.rectangle(crp_img, (x, y), (x + w, y + h), (31, 255, 31), 1)  # Green.

    # We can resize the frame here if want to. I've left it as is.
    img_res = cv2.resize(frame, (int(frame.shape[1]/1), int(frame.shape[0]/1)))

    # Show the individual frame.
    cv2.imshow('Playback', img_res)
    # Wait for 17 milliseconds, the go to the next statement.
    # For a 60fps video, choose a time delay of 16.67ms.
    # For a 24fps (23.997fps) video, choose a time delay of 41.68ms.
    # For a 30fps video, choose a time delay of 33.33ms.
    key = cv2.waitKey(17)

    # Break this loop if 'Q' is pressed on the keyboard.
    # WARNING! Video will continue to playback until the break condition is detected.
    if key == ord('q'):
        break
# Print 'a' i.e, how many frames did we display.
print(a)

# Unbind the video. Release the video capture device in our case.
vid.release()

# Destroy all created windows that were used to display the video.
cv2.destroyAllWindows()
