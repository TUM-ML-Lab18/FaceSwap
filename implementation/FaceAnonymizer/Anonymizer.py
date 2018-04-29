import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

from FaceAnonymizer.Trainer import Trainer
from FaceExtractor import FaceExtractor


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
        self.model = Trainer(None, None)
        self.model.load_model(self.model_folder)

        # use extractor und transform later get correct input for network
        self.extractor = FaceExtractor()
        self.to_tensor = Compose([Resize((64, 64)), ToTensor()])
        self.from_tensor = Compose([ToPILImage()])

    def convert_images(self):
        """
        Converts the images and saves them.
        """
        for idx, (img, _) in enumerate(self.image_dataset):
            # extract information from image
            extracted_information = self.extractor(np.array(img))
            network_input = Image.fromarray(extracted_information.image)
            border = extracted_information.bounding_box
            network_input = self.to_tensor(network_input)
            extracted_width, extracted_height = border.right - border.left, border.bottom - border.top

            # get network output
            network_output = self.model.anonymize(network_input.unsqueeze(0).cuda()).squeeze(0)
            # get it back to the cpu and get the data
            network_output = network_output.cpu().detach()

            # resize it to original dimensions
            network_output = Resize((extracted_height, extracted_width))(self.from_tensor(network_output))

            # put the output onto the original image
            img.paste(network_output, (border.left, border.top))

            # save image
            img.save(self.output_folder.__str__() + "/" + str(idx) + ".jpg", "JPEG")
