import cv2
import glob
import os

from util.constants import emotions, face_img_width, face_img_height

dir_path = os.path.dirname(os.path.realpath(__file__))

faceDet = cv2.CascadeClassifier(dir_path + "/face_models/haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier(dir_path + "/face_models/haarcascade_frontalface_alt.xml")
faceDet_three = cv2.CascadeClassifier(dir_path + "/face_models/haarcascade_frontalface_alt2.xml")
faceDet_four = cv2.CascadeClassifier(dir_path + "/face_models/haarcascade_frontalface_alt_tree.xml")


def make_dirs():
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    for emotion in emotions:
        emotion_dir = os.path.join("dataset", emotion)
        if not os.path.exists(emotion_dir):
            os.mkdir(emotion_dir)


def detect_faces_in_dataset(emotion):
    files = glob.glob("sorted_set/%s/*" % emotion)  # Get list of all images with emotion
    print("Detecting faces in %i images for %s" % (len(files), emotion))
    for file_number, f in enumerate(files):
        if file_number % (int(len(files) / 5)) == 0:
            print("%f percent done." % (file_number / len(files)))

        face_features, gray = detect_faces(f)

        if face_features is None:
            # no faces
            continue

        # Cut and save face
        for (x, y, w, h) in face_features:  # get coordinates and size of rectangle containing face
            print("face found in file: %s of size (%f, %f) at (%i, %i)" % (f, w, h, x, y))
            face_gray = gray[y:y + h, x:x + w]  # Cut the frame to size

            try:
                # Resize face so all images have same size
                out = cv2.resize(face_gray, (face_img_width, face_img_height))
                outfile = "dataset/%s/%s.jpg" % (emotion, file_number)
                print("Saving to " + outfile)
                cv2.imwrite(outfile, out)  # Write image
            except Exception as ex:
                print("Couldn't save file")
                print(ex)


def detect_faces(img_file):
    frame = cv2.imread(img_file)  # Open image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    # Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

    # Go over detected faces, stop at first detected face, return None if no face.
    if len(face) == 1:
        face_features = face
    elif len(face_two) == 1:
        face_features = face_two
    elif len(face_three) == 1:
        face_features = face_three
    elif len(face_four) == 1:
        face_features = face_four
    else:
        face_features = None

    return face_features, gray


def get_all_faces(img_file):
    faces_features, gray = detect_faces(img_file)

    if faces_features is None:
        return None

    faces = []

    for (x, y, w, h) in faces_features:  # get coordinates and size of rectangle containing face
        face_gray = gray[y:y + h, x:x + w]  # Cut the frame to size
        # Resize face so all images have same size
        out_face = cv2.resize(face_gray, (face_img_width, face_img_height))
        faces.append((out_face, (x, y, w, h)))

    return faces


if __name__ == "__main__":
    make_dirs()
    for emotion in emotions:
        detect_faces_in_dataset(emotion)
