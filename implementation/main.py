from FaceAnonymizer.Trainer import Trainer
from Configuration.config_model import current_config

if __name__ == '__main__':
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    trainer = Trainer(None, current_config)

    trainer.train()
