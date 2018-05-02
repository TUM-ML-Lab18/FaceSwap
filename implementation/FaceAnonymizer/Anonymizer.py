import cv2
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

from FaceAnonymizer.Trainer import Trainer
from Preprocessor.FaceExtractor import FaceExtractor


class Anonymizer:
    def __init__(self, images_folder: str, output_folder: str, model_folder: str) -> None:
        """
        :param images_folder: Path to images folder.
        :param output_folder: Path to output folder.
        :param model_folder: Path to models folder.
        """
        self.output_folder = Path(output_folder)
        self.images_folder = Path(images_folder)
        self.model_folder = Path(model_folder)

        shutil.rmtree(self.output_folder)
        self.output_folder.mkdir()

        # load dataset with ImagesFolder class -> PIL images
        self.image_dataset = ImageFolder(self.images_folder.__str__())
        self.model = Trainer(None)
        self.model.load_model(self.model_folder)

        # use extractor und transform later get correct input for network
        self.extractor = FaceExtractor()
        self.to_tensor = Compose([Resize((64, 64)), ToTensor()])
        self.from_tensor = Compose([ToPILImage()])

    def convert_images(self):
        """
        Converts the images and saves them.
        """
        for idx, (oringinal_img, _) in enumerate(self.image_dataset):
            # extract information from image
            extracted_information = self.extractor(oringinal_img)

            network_input = extracted_information.image
            network_input.save(self.output_folder.__str__() + "/" + str(idx) + "_network_input.jpg", "JPEG")
            border = extracted_information.bounding_box
            network_input = self.to_tensor(network_input)
            extracted_width, extracted_height = border.right - border.left, border.bottom - border.top

            # get network output
            network_output = self.model.anonymize(network_input.unsqueeze(0).cuda()).squeeze(0)
            # get it back to the cpu and get the data
            network_output = self.from_tensor(network_output.cpu().detach())
            network_output.save(self.output_folder.__str__() + "/" + str(idx) + "_network_output.jpg", "JPEG")

            # resize it to original dimensions
            # i dont know why height and width have to be switched
            network_output = Resize((extracted_height, extracted_width))(network_output)

            # get flat list of landmark tuple points
            landmarks = np.asarray([item for sublist in [x for (_, x) in
                                                         extracted_information.face_landmarks.items()] for item in
                                    sublist])

            # convert output to np array
            output_np = np.asarray(network_output)
            # use this feature map to copy the facial part of the output onto the original image
            feature_mask = np.zeros((oringinal_img.height, oringinal_img.width))
            head_features = cv2.convexHull(landmarks)
            cv2.fillConvexPoly(feature_mask, head_features, 1)
            feature_mask = feature_mask.astype(np.bool)

            # generate image that is as big as the original and copy the output on the correct position
            out_face = np.zeros((oringinal_img.height, oringinal_img.width, 3), dtype=np.uint8)
            out_face[border.top:border.bottom, border.left: border.right, :] = output_np

            # now use feature map and only copy the face onto the original image
            oringinal_img = np.array(oringinal_img).copy()
            oringinal_img[feature_mask] = out_face[feature_mask]
            oringinal_img = Image.fromarray(oringinal_img)

            # save image
            oringinal_img.save(self.output_folder.__str__() + "/" + str(idx) + ".jpg", "JPEG")
