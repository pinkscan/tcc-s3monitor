import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "best_model.pth2"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_model_structure():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    return model


def load_model():
    model = create_model_structure()

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model
