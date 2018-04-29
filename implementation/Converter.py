import os
import torch

import cv2

import numpy as np
from torchvision.transforms import ToTensor

from FaceAnonymizer.Trainer import Trainer
from Preprocessor.Dataset import DatasetPerson
from Preprocessor.Preprocessor import Preprocessor
from config import TRUMP_CAGE_BASE, CONVERTER_INPUT


def tensor2img(output):
    inv_idx = torch.arange(output.size(0) - 1, -1, -1).long()
    output = output[inv_idx]
    return output


class Converter:
    def __init__(self, images_folder, model_folder):
        self.folder = images_folder
        self.data = DatasetPerson(self.folder, Preprocessor(), transform=False, warp_faces=False,
                                  size_multiplicator=1, convertion_dataset=True)
        self.model = Trainer(None, None)
        self.model.load_model(model_folder)

    def convert_images(self):
        for idx, (real_image, (left, top, right, bottom)) in enumerate(zip(self.data.real_images, self.data.borders)):
            # tensor = tensor2img(tensor)
            _, tensor = self.data.__getitem__(idx)
            input = tensor.detach().cpu().numpy()
            input = (input * 255.0).astype(np.uint8)
            input = input.transpose(1, 2, 0)
            input = input[:, :, ::-1].copy()
            cv2.imwrite(os.path.join(self.folder + "/converted", str(idx) + "_input.jpg"), input)

            new_image = self.model.anonymize(tensor.unsqueeze(0).cuda()).squeeze(0)
            new_image = new_image.data.cpu().numpy()
            new_image = (new_image * 255.0).astype(np.uint8)
            new_image = new_image.transpose(1, 2, 0)
            new_image = cv2.resize(new_image, (right - left, bottom - top))
            new_image = new_image[:, :, ::-1]
            cv2.imwrite(os.path.join(self.folder + "/converted", str(idx) + "_output.jpg"), new_image)
            for i in range(left, right):
                for j in range(top, bottom):
                    real_image[j, i, :] = new_image[j - top, i - left, :]

            cv2.imwrite(os.path.join(self.folder + "/converted", str(idx) + ".jpg"), real_image)


if __name__ == '__main__':
    c = Converter(TRUMP_CAGE_BASE + CONVERTER_INPUT, "./logs/2018-04-28 17:17:02.447529/model__20180428_224553")
    c.convert_images()
