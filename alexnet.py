from sequential import Sequential
import pylayer as L

class AlexNet(Sequential):
    def __init__(self, output_classes=1000):
        super().__init__([
            L.Conv2d(3, 96, (11, 11), padding=0, stride=4),
            L.ReLU(),
            L.MaxPool2d((3, 3), padding=0, stride=2),
            L.Conv2d(96, 256, (5, 5), padding=2, stride=1),
            L.ReLU(),
            L.MaxPool2d((3, 3), padding=0, stride=2),
            L.Conv2d(256, 384, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.Conv2d(384, 384, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.Conv2d(384, 256, (3, 3), padding=1, stride=1),
            L.ReLU(),
            L.MaxPool2d((3, 3), padding=0, stride=2),
            L.Flatten(),
            L.Linear(5*5*256, 4096),
            L.ReLU(),
            L.Linear(5*5*256, 4096),
            L.ReLU(),
            L.Linear(4096, output_classes),
        ])