# IBot para Classificação de Atributos de Roupas

Iniciação Científica

Orientadores: Alceu Britto e Arlete Beuren

## O que é esse projeto?

Esse projeto usa o IBot para classificar atributos de roupas, como cor e textura sem ficar retreinando o modelo. A ideia é extrair features de diferentes blocos do modelo e ver qual bloco é melhor para cada tipo de atributo.

## Como funciona?

O IBot usa um Vision Transformer com 12 blocos. Cada bloco "enxerga" a imagem de um jeito diferente:

- Blocos iniciais (0-3): Vê detalhes básicos, como bordas e cores
- Blocos intermediários (4-7): Vê padrões simples como texturas e formas
- Blocos finais (8-11): Vê padrões mais complexos e semântica do objeto

A gente extrai as features de cada bloco e usa um k-NN para classificar. Assim dá pra comparar qual bloco funciona melhor pra cada tarefa e não precisa ficar retreinando o modelo.

## Dataset

O dataset tem 2000 imagens de roupas divididas em dois grupos. Em cor, são 1000 imagens com 10 classes: amarillo, azul, cafe, gris, morado, naranjo, negro, rojo, rosado e verde. Para textura, são 1000 imagens com 10 classes: argyle, cuadros, flores, lentejuelas, leopardo, paisley, pata_de_gallo, plano, polka e rayas.

As imagens foram redimensionadas para 224x224 e normalizadas com os valores do ImageNet.

## Protocolo Experimental

### Modelo

- IBot ViT-S/16 pré-treinado no ImageNet-1K com self-supervised learning
- Pesos do repositório oficial: ByteDance/ibot

### Extração de Features

- Features extraídas dos 12 blocos usando hooks
- CLS token para representar cada bloco
- Dimensão: 384 para todos os blocos

### Classificação

- Algoritmo: k-NN com k=5, distância euclidiana
- Validação: 5-fold Stratified Cross-Validation
- Em cada fold: 80% treino, 20% teste

### Métricas

- Acurácia média em %
- Desvio padrão entre os folds
- F1-Score

## Como rodar

### 1. Entrar no projeto
```bash
cd project
```

### 2. Criar ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

### 4. Baixar o modelo se não tiver
```bash
wget -O checkpoint_ibot_vits16.pth "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth"
```

### 5. Rodar o projeto
```bash
python3 main.py
```

## Requisitos

- Python 3
- GPU com CUDA
- ~4GB de RAM

## Referências

- [IBot: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832)
- [Repositório Oficial IBot - ByteDance](https://github.com/bytedance/ibot)
