from torch.utils.data import Dataset, DataLoader
import torch


class FTIRDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data 
        self.labels = labels 

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data = torch.randn(200, 1, 4000)
    labels = torch.randn(200)

    dataset = FTIRDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    for batch in dataloader: 
        s, l = batch
        print(f"Signal shape: {s.shape}")
        print(f"Labels shape: {l.shape}")