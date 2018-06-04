from pathlib import Path

import face_recognition, requests, io
import numpy as np
from PIL import Image

from FaceAnonymizer.Anonymizer import Anonymizer
from configuration.run_config import current_config
from configuration.evaluation_config import standard_conf


class Evaluator:

    @staticmethod
    def evaluate_model(model_folder='model', image_folder='/nfs/students/summer-term-2018/project_2/test/',
                       output_path='/nfs/students/summer-term-2018/project_2/test_alex/'):
        """
        Evaluates a model by comparing input images with output images
        :param model_folder: path to the model to evaluate
        :param image_folder: path images used to evaluate the model
        :param output_path: path where anonymized images should get stored
        :return: list of distances
        """
        image_folder = Path(image_folder)
        output_path = Path(output_path)
        anonymizer = Anonymizer(model_folder=model_folder,
                                model=current_config['model'],
                                config=current_config)
        print("The authors of the package recommend 0.6 as max distance for the same person.")
        distances = []
        for image_file in image_folder.iterdir():
            if image_file.is_dir():
                continue

            print('#' * 10)
            print('Processing image:', image_file.name)

            input_image = Image.open(image_file)

            anonymized_image = anonymizer(input_image)
            anonymized_image.save(output_path / ('anonymized_' + image_file.name.__str__()))

            distances.append(Evaluator.evaluate_image_pair(input_image, anonymized_image))
            print('Current image distance:', distances[-1])
        return distances

    @staticmethod
    def evaluate_image_pair(img1, img2, alpha=1, beta=1, treshold=0.6):
        """
        computes distances between img1 and img2
        :param img1: a single image
        :param img2: a single image
        :param beta
        :param alpha
        :param treshold
        :return: distance of images
        """
        similarity_score = Evaluator.get_similarity_score(img1, img2)
        print(f"Similarity score: {similarity_score}")
        emotion_score = Evaluator.get_emotion_score(img1, img2)
        print(f"Emotion score: {emotion_score_score}")

        score = 1 / (1 + np.exp(alpha * emotion_score - beta * (similarity_score - treshold)))

        return score

    @staticmethod
    def get_emotion_score(img1, img2):
        # create binary objects
        io1, io2 = io.BytesIO(), io.BytesIO()
        img1.save(io1, format='JPEG')
        img2.save(io2, format='JPEG')

        headers = {'Ocp-Apim-Subscription-Key': standard_conf['apiKey'], 'Content-Type': 'application/octet-stream'}

        r1 = requests.post(standard_conf['url'], headers=headers, data=io1.getvalue()).json()[0]
        r2 = requests.post(standard_conf['url'], headers=headers, data=io2.getvalue()).json()[0]

        emotions1 = np.array(list(r1['faceAttributes']['emotion'].values()))
        emotions2 = np.array(list(r2['faceAttributes']['emotion'].values()))

        return ((emotions1 - emotions2) ** 2).mean()

    @staticmethod
    def get_similarity_score(img1, img2):
        enconding1 = [
            face_recognition.face_encodings(np.array(img1), known_face_locations=[(0, img1.size[0], img1.size[0], 0)])[
                0]]
        enconding2 = [
            face_recognition.face_encodings(np.array(img2), known_face_locations=[(0, img2.size[0], img2.size[0], 0)])[
                0]]
        dist = face_recognition.face_distance(np.array(enconding1), np.array(enconding2))[0]
        return dist
