import torch

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            #torch.nn.ReLU(),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=4, stride=4))
        #self.layer2 = torch.nn.Sequential(
        #    torch.nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=2),
        #    torch.nn.ReLU(),
        #    torch.nn.MaxPool2d(kernel_size=2, stride=2))
        #self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(7 * 7 * 10, 100)
        #self.fc1 = torch.nn.Linear(14 * 14 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out