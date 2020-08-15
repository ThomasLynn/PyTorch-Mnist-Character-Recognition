import torch

class ConvNet_Small(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Small, self).__init__()
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
        
class ConvNet_Medium(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Medium, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            #torch.nn.ReLU(),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 10, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(7 * 7 * 10, 100)
        #self.fc1 = torch.nn.Linear(14 * 14 * 10, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
class ConvNet_Big(torch.nn.Module):
    def __init__(self):
        super(ConvNet_Big, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 30, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(7 * 7 * 30, 1000)
        #self.fc1 = torch.nn.Linear(14 * 14 * 10, 100)
        self.fc2 = torch.nn.Linear(1000, 100)
        self.fc3 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out