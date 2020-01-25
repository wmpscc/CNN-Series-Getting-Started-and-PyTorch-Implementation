import torch
from torch import nn
from torchex import nn as exnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TeacherNet(nn.Module):
    def __init__(self, classes):
        super(TeacherNet, self).__init__()
        self.classes = classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            exnn.GlobalAvgPool2d(),
            exnn.Flatten(),
            nn.Linear(256, self.classes)
        )

    def forward(self, X):
        tmp = self.conv(X)
        return self.fc(tmp)


class StudentNet(nn.Module):
    def __init__(self, classes):
        super(StudentNet, self).__init__()
        self.classes = classes
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            exnn.GlobalAvgPool2d(),
            exnn.Flatten(),
            nn.Linear(256, self.classes)
        )

    def forward(self, X):
        tmp = self.conv(X)
        return self.fc(tmp)


class Student2Net(nn.Module):
    def __init__(self, classes):
        super(Student2Net, self).__init__()
        self.classes = classes
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            exnn.GlobalAvgPool2d(),
            exnn.Flatten(),
            nn.Linear(256, self.classes)
        )

    def forward(self, X):
        tmp = self.conv(X)
        return self.fc(tmp)
