"""

For each photo
    processes for faces
    run classifer
    save face params (x, y, width, height) and emotion to csv

"""

from csv import DictWriter
import os
from classifier import get_classifier
from extract_faces import get_all_faces
from util.constants import emotions

test_image_path = "test_image.jpg"
out_csv_suffix = "faces.csv"

fieldnames = ["emotion", "confidence", "x", "y", "width", "height", "filename"]


def detect_and_save(image_path):
    out_csv_filename = image_path + out_csv_suffix
    print("Getting classifier")
    classifier = get_classifier()
    print("Detecting faces in " + image_path)
    faces = get_all_faces(image_path)

    if faces is None or len(faces) == 0:
        print("No faces in this photo!")
        return

    print("Predicting and saving results to " + out_csv_filename)

    if not os.path.exists(out_csv_filename):
        with open(out_csv_filename, 'w') as csv_file:
            writer = DictWriter(csv_file, fieldnames)
            writer.writeheader()

    with open(out_csv_filename, 'a') as csv_file:
        writer = DictWriter(csv_file, fieldnames)

        face_num = 1
        for face, (x, y, w, h) in faces:
            print("Face number %i" % (face_num))

            prediction, confidence = classifier.predict(face)
            emotion = emotions[prediction]
            print("%i confident that this face is %s" % (confidence, emotion))
            writer.writerow({
                "emotion": emotion,
                "confidence": confidence,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "filename": image_path,
            })
            face_num += 1


if __name__ == "__main__":
    detect_and_save(test_image_path)




