import shutil
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

from FaceAnonymizer.Trainer import Trainer


# todo pls remove if extractor works as intended
class Extractor(object):
    def extract(self, img):
        # network img, border, landmarks
        return [None, None, None]


class Anonymizer:
    ANONYMIZED_FOLDER = "anonymized"

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
        self.extractor = Extractor()
        self.to_tensor = Compose([Resize((64, 64)), ToTensor()])
        self.from_tensor = Compose([ToPILImage()])

    def convert_images(self):
        for idx, (img, _) in enumerate(self.image_dataset):
            network_input, border, landmarks = self.extractor.extract(img)

            # only for testing
            border = [img.width // 4, img.height // 4, img.width // 4 * 3, img.height // 4 * 3]
            network_input = img.crop(border)
            network_input = self.to_tensor(network_input)

            width, height = border[2] - border[0], border[3] - border[1]
            # get network output
            network_output = self.model.anonymize(network_input.unsqueeze(0).cuda()).squeeze(0)
            # get it back to the cpu and get the data
            network_output = network_output.cpu().detach()
            network_output = Resize((height, width))(self.from_tensor(network_output))

            img.paste(network_output, (border[0], border[1]))
            img.save(self.output_folder.__str__() + "/" + str(idx) + ".jpg", "JPEG")
