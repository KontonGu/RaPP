import os
import argparse

from torch_geometric.data import Dataset
from torch_geometric.data.lightning import LightningDataset
import torch
from torch_geometric import seed_everything
import pytorch_lightning as pl

from rapp_model import RaPPModel



class RaPPDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.data_dir = root
        self.data_items = sorted(os.listdir(root))
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return self.data_items
    
    @property
    def num_classes(self) -> int:
        return 1
    
    def process(self):
        pass
    
    def len(self):
        return len(self.data_items) - 1
    
    def get(self, idx):
        file_name = self.data_items[idx]
        file_path = os.path.join(self.data_dir, file_name)
        print(file_path)
        data = torch.load(file_path, weights_only=False)
        return data


def train(dataset_dir, epoch, pretrained=False):
    seed_everything(123)
    dataset = RaPPDataset(root=dataset_dir)
    dataset = dataset.shuffle()
    dataset_len = len(dataset)
    
    dataset_portion = [0.85, 0.1, 0.05]
    
    train_idx = int(dataset_portion[0] * dataset_len)
    val_idx = int(dataset_portion[1] * dataset_len + train_idx)
    test_idx = int(dataset_portion[2] * dataset_len + val_idx)
    
    train_dataset = dataset[:train_idx]
    val_dataset = dataset[train_idx:val_idx]
    test_dataset = dataset[val_idx:]
    
    
    
    dataload = LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size= 1,
    )
    
    
    model = RaPPModel()
    if pretrained:
        model = RaPPModel.load_from_checkpoint('./lightning_logs/version_0/checkpoints/epoch=199-step=9078000.ckpt')
    devices = torch.cuda.device_count()
    train_strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=6, mode='min') 
    trainer = pl.Trainer(strategy=train_strategy, accelerator='gpu', devices=devices,
                         max_epochs=epoch, callbacks=[checkpoint])
    
    
    trainer.fit(model, dataload)
    trainer.test(model, dataload)
    



def main():
    parser = argparse.ArgumentParser(description="The script to train the model.")
    parser.add_argument("--dataset_dir", type=str, default="../data/model_dataset", help="the directory to the dataset")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--pretrained", type=bool, default=False)
    args = parser.parse_args()
    train(args.dataset_dir, args.epoch, args.pretrained)
    


if __name__ == '__main__':
    main()