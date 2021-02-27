import torch

class Forward_1(torch.nn.Module):
    def __init__(self):
        super(Forward_1, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 5_000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(5_000, 3_000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(3_000, 10))
        
    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        #print(out.shape)
        return self.layer(out)

class Forward_2(torch.nn.Module):
    def __init__(self):
        super(Forward_2, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, 200, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1),
            torch.nn.Conv2d(200, 400, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1),
            torch.nn.Conv2d(400, 800, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 2),
            torch.nn.Conv2d(800, 1600, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 4 * 1600, 2_000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2_000, 1_000),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1_000, 10))
        #self.layer2 = torch.nn.Sequential())
        
    def forward(self, x):
        #print(x.shape)
        out = self.layer(x)
        #print(out.shape)
        #out = torch.nn.functional.dropout2d(out, p = 0.1)
        #out = out.reshape(out.size(0), -1)
        #print(out.shape)
        #out = self.layer2(out)
        #print(out.shape)
        return out

class Forward_3(torch.nn.Module):
    def __init__(self):
        super(Forward_3, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, 200, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1),
            torch.nn.Conv2d(200, 400, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 1),
            torch.nn.Conv2d(400, 800, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding = 2),
            torch.nn.Conv2d(800, 1600, kernel_size=3, stride=1, padding=1),
            torch.nn.Dropout2d(p = 0.1),
            torch.nn.LeakyReLU())
        self.layer2 = torch.nn.Transformer(d_model = 1600)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4*4*1600,10))
        self.poop = torch.rand((1500,16,1600))
        
    def forward(self, x):
        #print(x.shape)
        x = self.layer(x)
        x = x.reshape(x.size(0), 16, 1600)
        #print(out.shape)
        #out = torch.nn.functional.dropout2d(out, p = 0.1)
        print(x.shape,self.poop)
        
        x = self.layer2(self.poop,x)
        x = self.layer3(x)
        #print(out.shape)
        return x

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
        sizes = [600,800,1000]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, sizes[0], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(sizes[2], 10)
        #self.s1 = torch.nn.LeakyReLU()
        #self.fc2 = torch.nn.Linear(1_000, 10)
        
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
        #out = self.s1(out)
        #out = self.fc2(out)
        return out
        
class ConvNet_9(torch.nn.Module):
    def __init__(self):
        super(ConvNet_9, self).__init__()
        sizes = [400,500,600]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, sizes[0], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(sizes[2], 10)
        #self.s1 = torch.nn.LeakyReLU()
        #self.fc2 = torch.nn.Linear(1_000, 10)
        
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
        #out = self.s1(out)
        #out = self.fc2(out)
        return out
        
class ConvNet_10(torch.nn.Module):
    def __init__(self):
        super(ConvNet_10, self).__init__()
        sizes = [100,150,200]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, sizes[0], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding = 1))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU())
        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=4))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(sizes[2], 10)
        #self.s1 = torch.nn.LeakyReLU()
        #self.fc2 = torch.nn.Linear(1_000, 10)
        
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
        #out = self.s1(out)
        #out = self.fc2(out)
        return out
        
class ConvNet_11(torch.nn.Module):
    def __init__(self):
        super(ConvNet_11, self).__init__()
        #32x32
        self.cnnlayer = torch.nn.Sequential(
            torch.nn.Conv2d(1, 200, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(200, 400, kernel_size=3, stride=1, padding=1, groups = 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(400, 800, kernel_size=3, stride=1, padding=1, groups = 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(800, 1600, kernel_size=3, stride=1, padding=1, groups = 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(1600, 3200, kernel_size=3, stride=1, padding=1, groups = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(3200, 1600, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d())
        self.fc1 = torch.nn.Linear(1600, 10)
        
    def forward(self, x):
        out = self.cnnlayer(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out

class ConvNet_12(torch.nn.Module):
    def __init__(self):
        super(ConvNet_12, self).__init__()
        size = 100
        sizes = [size , int(size * 1.5), int(size * 1.5 ** 2)]
        print("sizes",sizes)
        self.layerstart = torch.nn.Conv2d(1, size, kernel_size=1)
        self.layers = []
        for i in range(10):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Conv2d(sizes[0], sizes[0] // 2, kernel_size = 1),
                torch.nn.Conv2d(sizes[0] // 2, sizes[0], kernel_size = 3, padding = 1),
                torch.nn.Dropout2d(p = 0.1),
                torch.nn.LeakyReLU()))
        self.layersmod = torch.nn.ModuleList(self.layers)
        
        self.layerpool = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[0], sizes[1], kernel_size = 1),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.LeakyReLU())
            
        self.layers2 = []
        for i in range(10):
            self.layers2.append(torch.nn.Sequential(
                torch.nn.Conv2d(sizes[1], sizes[1] // 2, kernel_size = 1),
                torch.nn.Conv2d(sizes[1] // 2, sizes[1], kernel_size = 3, padding = 1),
                torch.nn.Dropout2d(p = 0.1),
                torch.nn.LeakyReLU()))
        self.layers2mod = torch.nn.ModuleList(self.layers2)
        
        self.layerpool2 = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[1], sizes[2], kernel_size = 1),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.LeakyReLU())
            
        self.layers3 = []
        for i in range(10):
            self.layers3.append(torch.nn.Sequential(
                torch.nn.Conv2d(sizes[2], sizes[2] // 2, kernel_size = 1),
                torch.nn.Conv2d(sizes[2] // 2, sizes[2], kernel_size = 3, padding = 1),
                torch.nn.Dropout2d(p = 0.1),
                torch.nn.LeakyReLU()))
        self.layers3mod = torch.nn.ModuleList(self.layers3)
        self.layerend = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(7*7*sizes[2],10))
        
    def forward(self, x):
        x = self.layerstart(x)
        for w in self.layers:
            old = x
            x = old + w(x)
        x = self.layerpool(x)
        for w in self.layers2:
            old = x
            x = old + w(x)
        x = self.layerpool2(x)
        for w in self.layers3:
            old = x
            x = old + w(x)
        x = self.layerend(x)
        return x
        
class ConvNet_13(torch.nn.Module):
    def __init__(self):
        super(ConvNet_13, self).__init__()
        #32x32
        sizes = [1,200,400,600,800,1000]
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #16x16
            torch.nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #8x8
            torch.nn.Conv2d(sizes[2], sizes[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #4x4
            torch.nn.Conv2d(sizes[3], sizes[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #2x2
            torch.nn.Conv2d(sizes[4], sizes[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(sizes[5], sizes[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #1x1
            torch.nn.Dropout2d(),
            torch.nn.Flatten(),
            torch.nn.Linear(sizes[5], 10))
        
    def forward(self, x):
        out = self.layers(x)
        return out