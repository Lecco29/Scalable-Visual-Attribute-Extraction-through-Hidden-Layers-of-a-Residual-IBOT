# avaliacao com knn

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score


# essa funcao avalia features com knn e validacao cruzada (modo antigo)
def avaliarKNN(features, labels, k=5, nfolds=5):
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    
    resultados = {}
    
    for bloco, X in features.items():
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        
        # validacao cruzada
        cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
        acuracias = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        
        # f1 score
        knn.fit(X, y)
        predicao = knn.predict(X)
        f1 = f1_score(y, predicao, average='weighted')
        
        resultados[bloco] = {
            'accuracy_mean': acuracias.mean() * 100,
            'accuracy_std': acuracias.std() * 100,
            'f1_score': f1 * 100,
            'dim': X.shape[1]
        }
    
    return resultados


# essa funcao avalia features usando folds pre-definidos
def avaliarKNNComFolds(featuresTreino, labelsTreino, featuresTeste, labelsTeste, k=5):
    
    codificador = LabelEncoder()
    yTreino = codificador.fit_transform(labelsTreino)
    yTeste = codificador.transform(labelsTeste)
    
    resultados = {}
    
    for bloco in featuresTreino.keys():
        xTreino = featuresTreino[bloco]
        xTeste = featuresTeste[bloco]
        
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(xTreino, yTreino)
        
        predicao = knn.predict(xTeste)
        acc = accuracy_score(yTeste, predicao) * 100
        f1 = f1_score(yTeste, predicao, average='weighted') * 100
        
        resultados[bloco] = {
            'accuracy': acc,
            'f1_score': f1,
            'dim': xTreino.shape[1]
        }
    
    return resultados
