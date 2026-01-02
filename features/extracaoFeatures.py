# extracao de features do ibot

import numpy as np
import torch
from tqdm import tqdm


# essa funcao extrai features de todas imagens usando media dos patches (12 blocos)
def extrairFeatures(extrator, imagens, dispositivo, tamanhoBatch=64):
    
    todasFeatures = {f'block{i}': [] for i in range(12)}
    numBatches = (len(imagens) + tamanhoBatch - 1) // tamanhoBatch
    
    for i in tqdm(range(numBatches), desc="Extraindo"):
        inicio = i * tamanhoBatch
        fim = min((i + 1) * tamanhoBatch, len(imagens))
        lote = imagens[inicio:fim].to(dispositivo)
        
        # extrai features
        features = extrator.extrairFeatures(lote, aplicarGAP=True)
        
        for bloco, feat in features.items():
            todasFeatures[bloco].append(feat.numpy())
    
    # junta batches
    for bloco in todasFeatures:
        todasFeatures[bloco] = np.vstack(todasFeatures[bloco])
    
    return todasFeatures


# essa funcao extrai features usando CLS token 
def extrairFeaturesComCLS(extrator, imagens, dispositivo, tamanhoBatch=64):
    
    todasFeatures = {f'block{i}': [] for i in range(12)}
    numBatches = (len(imagens) + tamanhoBatch - 1) // tamanhoBatch
    
    for i in tqdm(range(numBatches), desc="Extraindo (CLS)"):
        inicio = i * tamanhoBatch
        fim = min((i + 1) * tamanhoBatch, len(imagens))
        lote = imagens[inicio:fim].to(dispositivo)
        
        # extrai features com CLS
        features = extrator.extrairFeaturesComCLS(lote)
        
        for bloco, feat in features.items():
            todasFeatures[bloco].append(feat.numpy())
    
    # junta batches
    for bloco in todasFeatures:
        todasFeatures[bloco] = np.vstack(todasFeatures[bloco])
    
    return todasFeatures
