from pathlib import Path
import torch
import torchvision.transforms as transforms
from FaceAnonymizer.StyleTransferTrainer import StyleTransferTrainer
from FaceAnonymizer.models.Encoder import Encoder
from PIL import Image

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_emotion = Encoder(input_dim=(3, 128, 128), latent_dim=1024, num_convblocks=5)
    model_emotion.load("")

    model_face = Encoder(input_dim=(3, 128, 128), latent_dim=1024, num_convblocks=5)
    model_face.load("")

    loader = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor()
    ])

    image = Image.open("...")
    image = loader(image).unsqueeze(0)
    image = image.to(device, torch.float)

    trainer = StyleTransferTrainer(model_emotion, model_face, image, 200, 0.5, 0.5)