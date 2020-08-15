import torch

class Perceptron_1(torch.nn.Module):
    def __init__(self):
        super(Perceptron_1, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 300)
        self.s1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(300, 40)
        self.s2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(40, 10)
        
    def forward(self, x):
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc2(out)
        return out

class ConvNet_1(torch.nn.Module):
    def __init__(self):
        super(ConvNet_1, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=4, stride=4))
        self.fc1 = torch.nn.Linear(7 * 7 * 10, 100)
        self.s1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        return out
        
class ConvNet_2(torch.nn.Module):
    def __init__(self):
        super(ConvNet_2, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 10, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(7 * 7 * 10, 100)
        self.s1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        return out
        
class ConvNet_3(torch.nn.Module):
    def __init__(self):
        super(ConvNet_3, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(15, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 25, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        self.drop_out = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(4 * 4 * 25, 1000)
        self.s1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(1000, 100)
        self.s2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc3(out)
        return out