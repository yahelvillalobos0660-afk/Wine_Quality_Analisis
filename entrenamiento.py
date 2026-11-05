import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, recall_score,
    f1_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
)

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_dataset(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filepath, sep = ";")

    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()

    if 'density' in df.columns:
        df = df.drop(columns = ['density'])


    df['target'] = (df['quality']>5).astype(int)

    x = df.drop(columns= ['quality', 'target'])
    y = df['target']

    return x, y

def build_preprocessor(feature_names: list)  -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_names)
        ]
    )     

def evaluate_all_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    preprocessor = build_preprocessor(X_train.columns.tolist())


    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('cfl', LogisticRegression(random_state=42, max_iter=1000))
    ])

    pipe_dt = Pipeline([
        ('preprocessor', preprocessor),
        ('cfl', DecisionTreeClassifier(random_state=42))
    ])

    pipe_knn = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', KNeighborsClassifier())
    ])
    
    pipe_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', SVC(random_state=42, probability=True))
    ])

    param_grid_svm = {
        'clf__C': [0.1,1,10],
        'clf__kernel': ['rbf', 'linear'],
        'clf__gamma': ['scale', 'auto'],
    }

    grid_svm = GridSearchCV(
        estimator=pipe_svm,
        param_grid=param_grid_svm,
        cv= 5,
        scoring='roc_auc',
        n_jobs=-1
    )

    print("\n--- Entrenando y optimizando SVM con GridSearchCV ---")
    grid_svm.fit(X_train, y_train)
    print(f"Mejor ROC-AUC en CV para SVM: {grid_svm.best_score_:.4f}")
    print(f"Mejores Hiperparámetros: {grid_svm.best_params_}")

    pipe_lr.fit(X_train, y_train)
    pipe_dt.fit(X_train, y_train)
    pipe_knn.fit(X_train, y_train)

    models = {
        'Regresion Logistica': pipe_lr,
        'Arbol de desicion': pipe_dt,
        'KNN': pipe_knn,
        'SVM(Optimizado)': grid_svm.best_estimator_
    }

    result_list = []
    fig, axes = plt.subplots(2,2, figsize=(12,10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else model.decision_function(X_test)



        result_list.append({
            'Modelo': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)

        })


        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Calidad <= 5', 'Calidad > 5'])
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d', colorbar=False)
        axes[idx].set_title(f'Matriz de Confusión: {name}')



    plt.tight_layout()
    plt.savefig('matrices_confusion.png')
    plt.close()
    print("\n[Imagen guardada]: matrices_confusion.png" )

    #Creamos el dataframe comparativo      
    df_results = pd.DataFrame(result_list) 
    df_results = df_results.sort_values(by=['ROC-AUC', 'F1-Score'], ascending = False).reset_index(drop= True)

    print("\n=== TABLA COMPARATIVA DE DESEMPEÑO (TEST SET) ===")
    print(df_results.to_string(index=False))

    best_svm = grid_svm.best_estimator_
    print("\n=== REPORTE DE CLASIFICACIÓN DETALLADO (MEJOR MODELO: SVM) ===")
    print(classification_report(y_test, best_svm.predict(X_test)))
    return best_svm
MODEL_PATH = './Best_wine_model.pkl'  # <-- Agrega esta línea
def predict_new_wine(wine_data: dict):
    """Carga el modelo guardado y predice la calidad de una nueva muestra de vino."""
    # 1. Cargar el pipeline serializado (incluye escalador y modelo)

    model = joblib.load(MODEL_PATH)
    
    # 2. Convertir el diccionario de datos a un DataFrame de Pandas
    df_new = pd.DataFrame([wine_data])
    
    # 3. Realizar predicción y calcular probabilidad
    prediction = model.predict(df_new)[0]
    
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(df_new)[0][1]
        print(f"Probabilidad de ser Buena Calidad: {probability:.2%}")
    
    label = "Buena Calidad" if prediction == 1 else "Baja/Normal Calidad"
    print(f"Resultado de la Predicción: {label} (Clase: {prediction})")
    
    return prediction

if __name__== "__main__":
        DATA_PATH = './data/winequality-white.csv'

        x, y = prepare_dataset(DATA_PATH)

        X_train, X_test, y_train, y_test = train_test_split(
            x,y, test_size=0.2, random_state=42, stratify=y
        )

        best_model = evaluate_all_models(X_train, y_train, X_test, y_test)


        import joblib
        #Guardamos el pipeline con el sescalador y el modelo SVM optimizado
        joblib.dump(best_model, 'Best_wine_model.pkl')
        print("\n[Modelo exportado]: best_wine_model.pkl")
        #pkl contiene tanto el StandardScaler ajustado como el modelo SVM. Cuando llegue un nuevo dato, no tendrás que escalarlo manualmente; el pipeline lo hace automáticamente.    
        nuevo_vino = {
        'fixed acidity': 7.0,
        'volatile acidity': 0.27,
        'citric acid': 0.36,
        'residual sugar': 2.07,
        'chlorides': 0.045,
        'free sulfur dioxide': 45.0,
        'total sulfur dioxide': 170.0,
        'density': 0.9914,
        'pH': 3.00,
        'sulphates': 0.45,
        'alcohol': 12.8
    }

        print("--- Evaluando nuevo vino ---")
        predict_new_wine(nuevo_vino)