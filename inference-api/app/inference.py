import torch
from .model_loader import DEVICE

def run_inference_rf(model_rf, features):
    """
    Executa a detecção de Câncer (Normal vs Cancer) usando Random Forest
    """
    if model_rf is None:
        return {"erro": "Modelo RF não carregado"}

    probas = model_rf.predict_proba(features)[0]
    # Assumindo classes: 0 -> Normal, 1 -> Cancer (conforme código Flask original)
    prob_normal = float(probas[0] * 100)
    prob_cancer = float(probas[1] * 100)

    classe = "CANCER" if prob_cancer > prob_normal else "NORMAL"

    return {
        "classe_detectada": classe,
        "prob_normal": round(prob_normal, 2),
        "prob_cancer": round(prob_cancer, 2)
    }

def run_inference_resnet(model_resnet, tensor):
    """
    Executa a classificação (Benigno vs Maligno) usando ResNet
    """
    if model_resnet is None:
        return {"erro": "Modelo ResNet não carregado"}

    with torch.no_grad():
        outputs = model_resnet(tensor.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)

        prob_benigno = float(probs[0][0] * 100)
        prob_maligno = float(probs[0][1] * 100)

        classe = "MALIGNO" if prob_maligno > prob_benigno else "BENIGNO"

        return {
            "subtipo": classe,
            "prob_benigno": round(prob_benigno, 2),
            "prob_maligno": round(prob_maligno, 2),
            "confianca_subtipo": round(max(prob_benigno, prob_maligno), 2)
        }
