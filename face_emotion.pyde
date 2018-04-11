from csv import DictReader
from glob import glob
import os

from emotions.facial_emotion.util.constants import fieldnames

# All processing globals
global nf, save, createImage, fill, ellipse, size, color, noStroke, colorMode, loadImage, image, background, HSB


def get_frame_faces(csv_filename):
    with open(csv_filename, "r") as csvfile:
        reader = DictReader(csvfile, fieldnames=fieldnames)
        next(reader, None)  # skip header
        faces = []
        for line in reader:
            line["width"] = float(line["width"])
            line["height"] = float(line["height"])
            line["x"] = float(line["x"])
            line["y"] = float(line["y"])
            # print(line)
            faces.append(Face(line))
    return faces


class Face(object):
    def __init__(self, face_data):
        self.face_data = face_data

    def __getattr__(self, item):
        return self.face_data[item]

    def __getitem__(self, item):
        return self.face_data[item]

    @staticmethod
    def get_chunk_from_image(face, src_img):
        img = createImage(int(face.width), int(face.height), RGB)
        img.loadPixels()

        for x in range(int(face.x), int(face.width + face.x)):
            for y in range(int(face.y), int(face.height + face.y)):
                index = int(face.width * (y - face.y) + (x - face.x))
                src_index = int(src_img.width * y + x)
                img.pixels[index] = src_img.pixels[src_index]

        img.updatePixels()
        return img


class FaceBlob(object):
    def __init__(self, c, face, frame):
        self.color = c
        self.face = face
        self.frame = frame

    def draw(self):
        fill(self.color)
        ellipse(float(self.face["x"]) + self.face["width"] / 2,
                float(self.face["y"]) + self.face["height"] / 2,
                float(self.face["width"]),
                float(self.face["height"])
                )
        # rect(float(self.face["x"]), float(self.face["y"]), float(self.face["width"]), float(self.face["height"]))


def setup():
    size(1024, 576)
    video_name = "trump-alt-left-right.mp4-bw.mp4"
    input_images = glob("./frames/%s/frame*.jpg" % video_name)
    input_images = sorted(input_images, key=lambda pth: int(os.path.basename(pth)[5:9]))
    colorMode(HSB)
    face_blobs = []

    colors = {
        "neutral": color(360, 0, 50, 10),
        "anger": color(0, 80, 100, 10),
        "sadness": color(180, 80, 100, 10),
        "happy": color(270, 80, 100, 10),
        "surprise": color(60, 80, 100, 10),
        "fear": color(120, 80, 100, 10),
        "contempt": color(90, 100, 50, 10)
    }

    for frame_number, img_path in enumerate(input_images):
        print("Processing frame %i: %i/%i" % (frame_number, frame_number + 1, len(input_images)))
        img = loadImage(img_path)
        background(0)
        # image(img, 0, 0)
        csv_filename = img_path + "faces.csv"
        if os.path.exists(csv_filename):
            faces = get_frame_faces(csv_filename)
            noStroke()

            if len(faces) >= 2:
                print("Found %i faces in frame %i!" % (len(faces), frame_number))

            for face in faces:
                face_blobs.append(FaceBlob(colors[face["emotion"]], face, frame_number))
                face_chunk = Face.get_chunk_from_image(face, img)
                image(face_chunk, face.x, face.y)

        for blob in face_blobs:
            if frame_number > blob.frame + 15:  # let all blobs stay for half a second
                face_blobs.remove(blob)
            else:
                blob.draw()

        save("outframes/%s/frame%s.jpg" % (video_name, nf(frame_number, 4)))

    exit()
