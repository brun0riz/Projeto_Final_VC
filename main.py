from pathlib import Path
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm
import seaborn as sns

# Bibliotecas do scikit-learn (Aula 9)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # <-- Adicionado para o 2º modelo
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

path_root = Path(r"C:\Users\snriz\Documents\VC\pf\dataset")

# PP e seg
def preprocessar_imagem(caminho_imagem):
    img_bgr = cv2.imread(str(caminho_imagem))
    if img_bgr is None:
        raise ValueError(f"Erro ao carregar imagem: {caminho_imagem}")
    img_suavizada = cv2.GaussianBlur(img_bgr, (7, 7), 0)
    return img_bgr, img_suavizada

def segmentar_fruta(img_suavizada):
    gray = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    mask_limpa = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fechamento)
    
    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_limpa = cv2.morphologyEx(mask_limpa, cv2.MORPH_OPEN, kernel_abertura)
    
    contornos, _ = cv2.findContours(mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        maior_contorno = max(contornos, key=cv2.contourArea)
        mask_final = np.zeros_like(mask_limpa)
        cv2.drawContours(mask_final, [maior_contorno], -1, 255, thickness=cv2.FILLED)
        return mask_final, maior_contorno
    else:
        return mask_limpa, None

# ext de features
def extrair_features(img_bgr, mask, contorno):
    features = {}
    
    # 1. Forma
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    features['area'] = area
    features['perimetro'] = perimetro
    features['circularidade'] = 4 * np.pi * area / (perimetro**2) if perimetro > 0 else 0
    
    x, y, w, h = cv2.boundingRect(contorno)
    features['aspect_ratio'] = w / h if h > 0 else 0
    
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()
    for i in range(7):
        features[f'hu_{i}'] = hu[i]
    
    # 2. Cor (HSV)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    media_cor, desvio_cor = cv2.meanStdDev(hsv, mask=mask)
    features['h_media'] = media_cor[0][0]
    features['s_media'] = media_cor[1][0]
    features['v_media'] = media_cor[2][0]
    features['h_desvio'] = desvio_cor[0][0]
    
    # 3. Textura (GLCM) - Recorte
    img_recorte = cv2.cvtColor(img_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(img_recorte, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    features['glcm_contraste'] = graycoprops(glcm, 'contrast').mean()
    features['glcm_homogeneidade'] = graycoprops(glcm, 'homogeneity').mean()
    features['glcm_energia'] = graycoprops(glcm, 'energy').mean()
    features['glcm_correlacao'] = graycoprops(glcm, 'correlation').mean()
    
    return features

def processar_dataset_inteiro(path_origem, nome_csv_saida):
    dados = []
    imagens = list(path_origem.rglob("*.[jp][pn]*[g]"))
    print(f"Iniciando extração em {len(imagens)} imagens...")
    
    for img_path in tqdm(imagens, desc="Extraindo Features"):
        try:
            classe = re.sub(r"\d+", "", img_path.stem)
            img_bgr, img_suavizada = preprocessar_imagem(img_path)
            mask, contorno = segmentar_fruta(img_suavizada)
            
            if contorno is not None:
                features = extrair_features(img_bgr, mask, contorno)
                features['classe'] = classe
                dados.append(features)
        except Exception:
            pass 

    df = pd.DataFrame(dados)
    df.to_csv(nome_csv_saida, index=False)
    print(f"\nDataset salvo em: {nome_csv_saida}")
    return df

# analise de features
def gerar_boxplots_features(caminho_csv):
    print("\nGerando boxplots para análise de features...")
    df = pd.read_csv(caminho_csv)
    
    # Selecionamos 3 features representativas para o relatório (uma de cor, textura e forma)
    features_analise = ['h_media', 'glcm_contraste', 'circularidade']
    
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features_analise, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x='classe', y=feature, data=df, palette='Set2')
        plt.title(f'Boxplot: {feature}')
    
    plt.tight_layout()
    plt.show()

# train
def treinar_e_avaliar_modelo(caminho_csv):
    print("\nCarregando dados para treinamento...")
    df = pd.read_csv(caminho_csv)
    
    X = df.drop(columns=['classe'])
    y_raw = df['classe']
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nTreinando Modelo 1: Random Forest...")
    pipeline_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=10)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline_rf.fit(X_train, y_train)
    y_pred_rf = pipeline_rf.predict(X_test)
    
    print("\nRelatório de Classificação - Random Forest:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusão - Random Forest')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.show()

    # Ranking de features (Apenas Random Forest permite extrair isso facilmente)
    mascara_features = pipeline_rf.named_steps['selector'].get_support()
    features_selecionadas = X.columns[mascara_features]
    importancias = pipeline_rf.named_steps['classifier'].feature_importances_
    
    df_importancia = pd.DataFrame({'Feature': features_selecionadas, 'Importancia': importancias})
    print("\nRanking de Importância das Features (Random Forest):")
    print(df_importancia.sort_values(by='Importancia', ascending=False).head(5))

    print("\nTreinando Modelo 2: SVM (Support Vector Machine)...")
    pipeline_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=10)),
        ('classifier', SVC(kernel='rbf', random_state=42))
    ])
    
    pipeline_svm.fit(X_train, y_train)
    y_pred_svm = pipeline_svm.predict(X_test)
    
    print("\nRelatório de Classificação - SVM:")
    print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusão - SVM')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.show()


path_train = path_root / "train"
caminho_csv = path_root / "dataset_frutas_treino.csv"

# só roda uma vez, se for usar tuirar dnv
# processar_dataset_inteiro(path_train, caminho_csv)

gerar_boxplots_features(caminho_csv)

treinar_e_avaliar_modelo(caminho_csv)