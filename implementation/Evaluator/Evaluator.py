from pathlib import Path

import face_recognition
import json
import io
import numpy as np
import requests
from PIL import Image
from PIL.Image import BICUBIC
from torchvision.transforms import ToPILImage

from Configuration.config_evaluation import standard_conf
from Preprocessor.FaceExtractor import FaceExtractor


class Evaluator:

    @staticmethod
    def evaluate_model(config, model_folder, image_folder, output_path, save_json=True):
        """
        Evaluates a model by comparing input images with output images
        :param config: the model configuration
        :param model_folder: folder of the saved model
        :param image_folder: path images used to evaluate the model
        :param output_path: path where anonymized images should be stored
        :return: list of distances
        """

        image_folder = Path(image_folder)
        output_path = Path(output_path)
        model = config.model(**config.model_params)
        model.load_model(Path(model_folder))
        extractor = FaceExtractor(margin=0.05, mask_factor=10)

        print("The authors of the package recommend 0.6 as max distance for the same person.")
        scores = {}
        for image_file in image_folder.iterdir():
            if image_file.is_dir():
                continue

            print('#' * 10)
            print('Processing image:', image_file.name)

            input_image = Image.open(image_file)
            extracted_face, extracted_info = extractor(input_image)
            if extracted_face is None:
                print('Face could not be extracted')
                continue

            face_out = model.anonymize(extracted_face, extracted_info).squeeze(0)
            face_out = ToPILImage()(face_out.cpu().detach())
            face_out = face_out.resize(extracted_face.size, resample=BICUBIC)

            try:
                face_out.save(output_path / ('anonymized_' + image_file.name.__str__()))
                score, sim, emo = Evaluator.evaluate_image_pair(extracted_face, face_out)
                scores[image_file] = {'score': score, 'sim': sim, 'emo': emo, 'img': str(image_file.name)}
            except Exception as ex:
                print(ex)
                continue

            print('Current image score:', scores[image_file])

        if save_json:
            with open(output_path / 'scores.json', 'w') as f:
                json.dump(scores, f)

        return scores

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
        emotion_score = Evaluator.get_emotion_score(img1, img2)

        score = 1 / (1 + np.exp(alpha * emotion_score - beta * (similarity_score - treshold)))

        return score, similarity_score, emotion_score

    @staticmethod
    def get_emotion_score(img1, img2):
        # create binary objects
        io1, io2 = io.BytesIO(), io.BytesIO()
        img1.save(io1, format='JPEG')
        img2.save(io2, format='JPEG')

        headers = {'Ocp-Apim-Subscription-Key': standard_conf['apiKey'], 'Content-Type': 'application/octet-stream'}

        r1 = requests.post(standard_conf['url'] + 'detect?returnFaceAttributes=emotion', headers=headers, data=io1.getvalue()).json()[0]
        r2 = requests.post(standard_conf['url'] + 'detect?returnFaceAttributes=emotion', headers=headers, data=io2.getvalue()).json()[0]

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

    @staticmethod
    def get_api_similarity_score(img1, img2):
        io1, io2 = io.BytesIO(), io.BytesIO()
        img1.save(io1, format='JPEG')
        img2.save(io2, format='JPEG')

        headers = {'Ocp-Apim-Subscription-Key': standard_conf['apiKey'], 'Content-Type': 'application/octet-stream'}
        id1 = requests.post(standard_conf['url'] + 'detect', headers=headers, data=io1.getvalue()).json()[0]['faceId']
        id2 = requests.post(standard_conf['url'] + 'detect', headers=headers, data=io2.getvalue()).json()[0]['faceId']

        headers = {'Ocp-Apim-Subscription-Key': standard_conf['apiKey'], 'Content-Type': 'application/json'}
        req = {'faceId1': id1, 'faceId2': id2}
        r = requests.post(standard_conf['url'] + 'verify', headers=headers, json=req).json()

        return r['confidence']


