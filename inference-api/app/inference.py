import torch
from .model_loader import DEVICE

def run_inference(model, tensor):
    with torch.no_grad():
        outputs = model(tensor.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)

        prob_benigno = float(probs[0][0] * 100)
        prob_maligno = float(probs[0][1] * 100)

        classe = "MALIGNO" if prob_maligno > prob_benigno else "BENIGNO"

        return {
            "classe": classe,
            "prob_benigno": round(prob_benigno, 2),
            "prob_maligno": round(prob_maligno, 2),
            "confianca": round(max(prob_benigno, prob_maligno), 2)
        }
