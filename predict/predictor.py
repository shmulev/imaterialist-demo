from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEVICE = 'cuda'


def predict(model, loader, device):
    outputs = []
    tags = None
    for i_b, batch in enumerate(tqdm(loader)):
        tags, output = model.one_shot_predict(batch, device)
        outputs.append(output)
    return pd.DataFrame(data=np.concatenate(outputs), columns=tags)


def predict_mask(image_numpy, model, activation=None, device=DEVICE):
    x_tensor = torch.from_numpy(image_numpy).to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pr_mask = model.forward(x_tensor)
    
    if activation is not None:
        pr_mask = activation(pr_mask)
        
    pr_mask = pr_mask.squeeze().cpu().numpy().round()
    return pr_mask
