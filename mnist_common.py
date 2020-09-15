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
        
class ConvNet_4(torch.nn.Module):
    def __init__(self):
        super(ConvNet_4, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(15, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 25, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 40, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        #self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(2 * 2 * 40, 1000)
        self.s1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(1000, 100)
        self.s2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(100, 10)
        self.sm = torch.nn.Softmax(1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        #out = self.drop_out(out)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc3(out)
        out = self.sm(out)
        return out
        
class ConvNet_5(torch.nn.Module):
    def __init__(self):
        super(ConvNet_5, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 30, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 50, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 100, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(2 * 2 * 100, 1000)
        self.s1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(1000, 100)
        self.s2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(100, 10)
        self.sm = torch.nn.Softmax(1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc3(out)
        out = self.sm(out)
        return out
        
class ConvNet_6(torch.nn.Module):
    def __init__(self):
        super(ConvNet_6, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 15, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(15, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 25, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 45, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(2 * 2 * 45, 400)
        self.s1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(400, 60)
        self.s2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(60, 10)
        self.sm = torch.nn.Softmax(1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc3(out)
        out = self.sm(out)
        return out
        
class ConvNet_7(torch.nn.Module):
    def __init__(self):
        super(ConvNet_7, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 40, kernel_size=7, stride=1, padding=3),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(40, 60, kernel_size=7, stride=1, padding=3),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(60, 90, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(90, 135, kernel_size=5, stride=1, padding=2),
            torch.nn.LeakyReLU())
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(4 * 4 * 135, 4_000)
        self.s1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(4_000, 600)
        self.s2 = torch.nn.LeakyReLU()
        self.fc3 = torch.nn.Linear(600, 10)
        self.sm = torch.nn.Softmax(1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        out = self.s2(out)
        out = self.fc3(out)
        out = self.sm(out)
        return out
        
class ConvNet_8(torch.nn.Module):
    def __init__(self):
        super(ConvNet_8, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 450, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(450, 600, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(600, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(1000, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(1000, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(1000, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(1000, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(1000, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(1000, 1000, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(2 * 2 * 1000, 1_000)
        self.s1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(1_000, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        old_out = out
        out = self.layer4(out)
        out = self.layer5(out)
        out += old_out
        old_out = out
        out = self.layer6(out)
        out = self.layer7(out)
        out += old_out
        out = self.layer8(out)
        out = self.layer9(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.s1(out)
        out = self.fc2(out)
        return out