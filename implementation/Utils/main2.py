import torch
import torchvision.transforms as transforms
from Utils.StyleTransferTrainer import StyleTransferTrainer
from Models.DeepFake.Encoder import Encoder
from PIL import Image

if __name__ == '__main__':

    device = torch.device("cpu")

    model_emotion = Encoder(input_dim=(3, 128, 128), latent_dim=1024, num_convblocks=5)
    model_emotion.load("./model/encoder.model")

    model_face = Encoder(input_dim=(3, 128, 128), latent_dim=1024, num_convblocks=5)
    model_face.load("./model/encoder.model")

    loader = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    image = Image.open("trump_first.jpg")
    image = loader(image).unsqueeze(0)
    image = image.to(device, torch.float)

    trainer = StyleTransferTrainer(model_emotion, model_face, image, 50, 0.5, 0.5)

    trainer.train()

    result = transforms.ToPILImage()(trainer.result_img.squeeze(0))
    result.save("result.jpg")