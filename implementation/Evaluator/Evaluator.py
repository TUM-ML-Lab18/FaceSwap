import face_recognition


class Evaluator:
    def evaluate_model(self):
        pass

    def evaluate_list_of_images(self, img1, img2):
        enconding1 = [face_recognition.face_encodings(img)[0] for img in img1]
        enconding2 = [face_recognition.face_encodings(img)[0] for img in img2]
        dist = face_recognition.face_distance(enconding1, enconding2)
        return dist
