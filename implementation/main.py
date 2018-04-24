from torch.autograd import Variable
from torchvision.transforms import transforms

from FaceAnonymizer.models.Encoder import Encoder
from FaceAnonymizer.models.Decoder import Decoder
from Preprocessor import Dataset
from Preprocessor.Dataset import DatasetPerson, Resize, ToTensor

if __name__ == '__main__':
    d = DatasetPerson(Dataset.PROCESSED_IMAGES_FOLDER, transform=transforms.Compose([Resize((64, 64)), ToTensor()]))
    img = Variable(d[0].unsqueeze(0))
    print(img.shape)
    e = Encoder((3, 64, 64), 1024, 1)
    d = Decoder(512)
    x = e(img)
    x = d(x)
    print(x.size())
