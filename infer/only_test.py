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

def test_only(test_dir, checkpoint_path):
    seed_everything(123)
    dataset_test = RaPPDataset(root=test_dir)
    
    dataload = LightningDataset(
        train_dataset=None,
        val_dataset=None,
        test_dataset=dataset_test,
        batch_size=1,
    )
    
    # Load the pre-trained model
    model = RaPPModel.load_from_checkpoint(checkpoint_path)
    
    devices = torch.cuda.device_count()
    trainer = pl.Trainer(accelerator='gpu', devices=devices)
    
    # Only run the test step
    trainer.test(model, dataload)

def main():
    parser = argparse.ArgumentParser(description="The script to test the model.")
    parser.add_argument("--test_dir", type=str, default="../../data/model_dataset_test", help="the directory to the test dataset")
    # /home/ubuntu/konton_ws/RaPP/variant_model/GAT_split_deeper/lightning_logs/version_1/checkpoints/epoch=85-step=3903540.ckpt
    # /home/ubuntu/konton_ws/RaPP/variant_model/GAT_split_deeper/lightning_logs/version_1/checkpoints/epoch=183-step=8351760.ckpt
    parser.add_argument("--checkpoint_path", type=str, default="/home/ubuntu/konton_ws/RaPP/variant_model/GAT_split_deeper/lightning_logs/version_1/checkpoints/epoch=85-step=3903540.ckpt", help="path to the model checkpoint")
    args = parser.parse_args()
    
    test_only(args.test_dir, args.checkpoint_path)

if __name__ == '__main__':
    main()