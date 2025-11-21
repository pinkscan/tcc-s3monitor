import torch
import torch.nn as nn
from torchvision import models
import joblib
import os

# Caminhos dos modelos
RESNET_MODEL_PATH = "best_model.pth2"
RF_MODEL_PATH = "rf_model.joblib"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_resnet_structure():
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

def load_models():
    models_dict = {}

    # 1. Carregar ResNet (Benigno vs Maligno)
    if os.path.exists(RESNET_MODEL_PATH):
        print(f"Carregando ResNet de {RESNET_MODEL_PATH}...")
        resnet = create_resnet_structure()
        state = torch.load(RESNET_MODEL_PATH, map_location=DEVICE)
        resnet.load_state_dict(state)
        resnet.to(DEVICE)
        resnet.eval()
        models_dict['resnet'] = resnet
    else:
        print(f"AVISO: Modelo ResNet não encontrado em {RESNET_MODEL_PATH}")
        models_dict['resnet'] = None

    # 2. Carregar Random Forest (Normal vs Cancer)
    if os.path.exists(RF_MODEL_PATH):
        print(f"Carregando Random Forest de {RF_MODEL_PATH}...")
        rf_model = joblib.load(RF_MODEL_PATH)
        models_dict['rf'] = rf_model
    else:
        print(f"AVISO: Modelo RF não encontrado em {RF_MODEL_PATH}")
        models_dict['rf'] = None

    return models_dict
