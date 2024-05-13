import pytorch_lightning as pl 
from net import Net
from torch.utils.data import DataLoader 
from custom_dataset.lungdataset import LungDataset
from torch.utils.data import Subset
import torch 
from pytorch_lightning.loggers import CSVLogger

vd_train = LungDataset(
        '/home/bai_gairui/multi_label/data/Chest_x_ray/images/', 
        '/home/bai_gairui/multi_label/data/Chest_x_ray/Data_Entry_2017_v2020.csv', 
        allow_no_finding=True,
    )


vd_train, vd_test = torch.utils.data.random_split(vd_train, [len(vd_train) - len(vd_train) // 4,  len(vd_train) // 4])


train_loader = DataLoader(vd_train, batch_size=64, shuffle=True, drop_last=True)
test_laoder = DataLoader(vd_test, batch_size=64, shuffle=True, drop_last=True)

info = '''
Train on Resnet of Pytorh resnet50(inner classifier)
'''

mode = "train"
info = "resnet18"
if mode == "train":

    model  = Net(info = info, num_classes=15)

    logger = CSVLogger("logs", name=f"train-{info}")
    trainer = pl.Trainer(accelerator="gpu", devices=[6], precision=16, max_epochs=20, logger=logger)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_laoder)

else:
    checkpint_path = '/home/bai_gairui/multi_label/lightning_logs/version_24/checkpoints/epoch=15-step=1872.ckpt'
    hparam_file = '/home/bai_gairui/multi_label/lightning_logs/version_24/hparams.yaml'
    model = Net.load_from_checkpoint(checkpint_path, hparams_file=hparam_file)
    logger = CSVLogger('logs', name="test")
    trainer = pl.Trainer(accelerator="gpu", devices=[7], precision=16, max_epochs=20, logger=logger)
    trainer.test(model, test_laoder)
