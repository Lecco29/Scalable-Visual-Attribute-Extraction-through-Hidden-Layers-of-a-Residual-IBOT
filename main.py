#!/usr/bin/env python3

import os
import sys
import torch
import pandas as pd
from datetime import datetime
from PIL import Image
from torchvision import transforms

# caminhos do projeto
PASTA_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PASTA_RAIZ)
sys.path.insert(0, os.path.join(PASTA_RAIZ, 'models'))
sys.path.insert(0, os.path.join(PASTA_RAIZ, 'features'))
sys.path.insert(0, os.path.join(PASTA_RAIZ, 'analysis'))

from carregadorIBot import criarExtrator
from extracaoFeatures import extrairFeaturesComCLS
from avaliacaoKNN import avaliarKNN


# essa funcao vai ler o csv e carregar as imagens
def carregarImagens(arquivoCsv, pastaImagens):

    dados = pd.read_csv(arquivoCsv)
    
    # redimensiona e normaliza as imagens
    transformacao = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    listaImagens = []
    listaLabels = []
    
    for i, linha in dados.iterrows():
        caminho = os.path.join(pastaImagens, linha['image_name'])
        if os.path.exists(caminho):
            try:
                img = Image.open(caminho).convert('RGB')
                listaImagens.append(transformacao(img))
                listaLabels.append(linha['label'])
            except:
                print(f"erro ao carregar: {caminho}")
    
    return torch.stack(listaImagens), listaLabels


# essa funcao roda o projeto completo
def rodarProjetoIBOT(nome, arquivoCsv, pastaImagens, modelo, device):
    
    print(f"\n[{nome}]")
    
    # carrega imagens
    print("carregando imagens...")
    imagens, labels = carregarImagens(arquivoCsv, pastaImagens)
    print(f"total: {len(imagens)} imagens, {len(set(labels))} classes")
    
    # extrai features
    print("extraindo features...")
    features = extrairFeaturesComCLS(modelo, imagens, device)
    
    # avalia com knn
    print("avaliando com knn...")
    resultados = avaliarKNN(features, labels)
    
    # mostra resultados
    print(f"\nresultados {nome}:")
    for i in range(12):
        bloco = f'block{i}'
        acc = resultados[bloco]['accuracy_mean']
        print(f"  {bloco}: {acc:.2f}%")
    
    return resultados


def main():
    print("IBot - Classificacao de Roupas")
    print(f"inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    # verifica se tem gpu
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("usando cpu")
    
    # carrega modelo
    print("\ncarregando ibot...")
    modelo = criarExtrator(modelo='vit_small', dispositivo=device)
    
    # pasta dos dados
    pastaDados = os.path.join(PASTA_RAIZ, 'data')
    
    # arquivos csv
    csvCor = os.path.join(pastaDados, 'labels_color.csv')
    csvTextura = os.path.join(pastaDados, 'labels_texture.csv')
    
    # verifica se existe
    if not os.path.exists(csvCor):
        print(f"erro: arquivo nao encontrado: {csvCor}")
        return
    
    # cor
    resultadoCor = rodarProjetoIBOT(
        "COR",
        csvCor,
        os.path.join(pastaDados, 'images', 'color'),
        modelo, device
    )
    
    # textura
    resultadoTextura = rodarProjetoIBOT(
        "TEXTURA",
        csvTextura,
        os.path.join(pastaDados, 'images', 'texture'),
        modelo, device
    )
    
    # salva resultados
    print("\nsalvando...")
    for nome, resultado in [('color', resultadoCor), ('texture', resultadoTextura)]:
        df = pd.DataFrame([{'bloco': b, **d} for b, d in resultado.items()])
        arquivo = os.path.join(PASTA_RAIZ, f'results_{nome}_ibot.csv')
        df.to_csv(arquivo, index=False)
        print(f"salvo: {arquivo}")
    
    print(f"\nfim: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
