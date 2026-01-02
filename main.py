#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
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
from avaliacaoKNN import avaliarKNNComFolds


# transformacao para as imagens
TRANSFORMACAO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


# essa funcao le o arquivo de fold e carrega as imagens
def carregarImagensDoFold(arquivoFold, pastaBase, tipoAtributo):
    
    listaImagens = []
    listaLabels = []
    
    with open(arquivoFold, 'r') as f:
        linhas = f.readlines()
    
    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        
        # formato: indice;classe;caminho_original
        partes = linha.split(';')
        if len(partes) < 3:
            continue
        
        classe = partes[1]
        caminhoOriginal = partes[2]
        
        # extrai info do caminho
        # /mnt/data/fabric/dataset_atributos/color/amarillo/6996.jpg
        partesPath = caminhoOriginal.split('/')
        subpasta = partesPath[-2]  # amarillo
        nomeArquivoOriginal = partesPath[-1]  # 6996.jpg
        
        # monta o nome no formato do projeto: color_amarillo_6996.jpg
        nomeArquivoLocal = f"{tipoAtributo}_{subpasta}_{nomeArquivoOriginal}"
        caminhoLocal = os.path.join(pastaBase, nomeArquivoLocal)
        
        if os.path.exists(caminhoLocal):
            try:
                img = Image.open(caminhoLocal).convert('RGB')
                listaImagens.append(TRANSFORMACAO(img))
                listaLabels.append(classe)
            except Exception as e:
                print(f"erro ao carregar: {caminhoLocal}")
    
    if len(listaImagens) == 0:
        return None, None
    
    return torch.stack(listaImagens), listaLabels


# essa funcao roda o projeto com os folds
def rodarProjetoComFolds(nome, pastaFolds, pastaImagens, modelo, device, tipoAtributo):
    
    print(f"\n[{nome}]")
    
    dadosPorBloco = {f'block{i}': {'acc': [], 'f1': [], 'dim': 384} for i in range(12)}
    
    for numFold in range(1, 6):
        print(f"\n  fold {numFold}/5:")
        
        # arquivos do fold
        arquivoTreino = os.path.join(pastaFolds, f'fold{numFold}-train.txt')
        arquivoTeste = os.path.join(pastaFolds, f'fold{numFold}-test.txt')
        
        if not os.path.exists(arquivoTreino) or not os.path.exists(arquivoTeste):
            print(f"    arquivos nao encontrados!")
            continue
        
        # carrega imagens de treino
        print(f"    carregando treino...")
        imagensTreino, labelsTreino = carregarImagensDoFold(arquivoTreino, pastaImagens, tipoAtributo)
        if imagensTreino is None:
            print(f"    erro ao carregar treino")
            continue
        print(f"    treino: {len(imagensTreino)} imagens")
        
        # carrega imagens de teste
        print(f"    carregando teste...")
        imagensTeste, labelsTeste = carregarImagensDoFold(arquivoTeste, pastaImagens, tipoAtributo)
        if imagensTeste is None:
            print(f"    erro ao carregar teste")
            continue
        print(f"    teste: {len(imagensTeste)} imagens")
        
        # extrai features
        print(f"    extraindo features treino...")
        featuresTreino = extrairFeaturesComCLS(modelo, imagensTreino, device)
        
        print(f"    extraindo features teste...")
        featuresTeste = extrairFeaturesComCLS(modelo, imagensTeste, device)
        
        # avalia com knn
        print(f"    avaliando com knn...")
        resultados = avaliarKNNComFolds(featuresTreino, labelsTreino, featuresTeste, labelsTeste)
        
        # guarda acuracias, f1 e dim
        for bloco, res in resultados.items():
            dadosPorBloco[bloco]['acc'].append(res['accuracy'])
            dadosPorBloco[bloco]['f1'].append(res['f1_score'])
            dadosPorBloco[bloco]['dim'] = res['dim']
        
        # mostra resultado do fold
        melhorBloco = max(resultados.keys(), key=lambda b: resultados[b]['accuracy'])
        print(f"    melhor: {melhorBloco} = {resultados[melhorBloco]['accuracy']:.2f}%")
    
    # calcula media e desvio padrao
    print(f"\nresultados finais {nome} (media 5 folds):")
    resultadoFinal = {}
    
    for i in range(12):
        bloco = f'block{i}'
        accs = dadosPorBloco[bloco]['acc']
        f1s = dadosPorBloco[bloco]['f1']
        dim = dadosPorBloco[bloco]['dim']
        if len(accs) > 0:
            mediaAcc = np.mean(accs)
            stdAcc = np.std(accs)
            mediaF1 = np.mean(f1s)
            resultadoFinal[bloco] = {
                'accuracy_mean': mediaAcc,
                'accuracy_std': stdAcc,
                'f1_score': mediaF1,
                'dim': dim
            }
            print(f"  {bloco}: {mediaAcc:.2f}% (+/- {stdAcc:.2f})")
    
    return resultadoFinal


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
    pastaProtocolo = os.path.join(pastaDados, 'Protocolo')
    
    # verifica se existe
    if not os.path.exists(pastaProtocolo):
        print(f"erro: pasta nao encontrada: {pastaProtocolo}")
        return
    
    # cor - usando folds do protocolo
    resultadoCor = rodarProjetoComFolds(
        "COR",
        os.path.join(pastaProtocolo, 'folds_color', 'folds'),
        os.path.join(pastaDados, 'images', 'color'),
        modelo, device,
        tipoAtributo='color'
    )
    
    # textura - usando folds do protocolo
    resultadoTextura = rodarProjetoComFolds(
        "TEXTURA",
        os.path.join(pastaProtocolo, 'folds_texture', 'folds'),
        os.path.join(pastaDados, 'images', 'texture'),
        modelo, device,
        tipoAtributo='texture'
    )
    
    # salva resultados
    print("\nsalvando...")
    for nome, resultado in [('resultados_cor', resultadoCor), ('resultados_textura', resultadoTextura)]:
        df = pd.DataFrame([{'bloco': b, **d} for b, d in resultado.items()])
        arquivo = os.path.join(PASTA_RAIZ, f'{nome}.csv')
        df.to_csv(arquivo, index=False)
        print(f"salvo: {arquivo}")
    
    print(f"\nfim: {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
