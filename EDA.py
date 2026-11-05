import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def cargar_datos(filepath:str)-> pd.DataFrame:
    #Cargar primero el archivo csv

    if not os.path.exists(filepath):

        raise FileExistsError(f"El archivo {filepath} no existe")
    df = pd.read_csv(filepath, sep = ";")
    print(f"---> Datos cargados exitosamente. Forma: {df.shape}")
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
   # Verificación de nulos
    df_cleaned = df.copy()

    null_count = df_cleaned.isnull().sum().sum()
    if null_count > 0:
        print(f"Se encontraron {null_count} valores nulos. Limpiando...")
        df_cleaned = df_cleaned.dropna()
    else:
        print("No se encontraron valores nulos.")

    duplicates = df_cleaned.duplicated().sum()
    print(f"Filas duplicadas encontradas: {duplicates}")
    if duplicates > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Duplicados eliminados. Nuevo tamaño: {df_cleaned.shape}")
        
    return df_cleaned    
# =============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

def plot_correlation_matrix(df: pd.DataFrame, output_path: str = "matriz_correlacion.png"):
    """Genera y guarda la matriz de correlación de Pearson."""
    corr_matrix = df.corr()
    
    print("\n--- Correlación respecto a la variable 'quality' ---")
    print(corr_matrix['quality'].sort_values(ascending=False))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=-1, vmax=1,
        linewidths=0.5
    )
    plt.title('Matriz de Correlación - Wine Quality', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\n[Imagen guardada]: {output_path}")

def plot_pairplot(df: pd.DataFrame, features: list, output_path: str = "matriz_dispersion.png"):
    """Genera y guarda la matriz de dispersión (pairplot)."""
    sns.pairplot(
        df[features],
        hue='quality',
        palette='viridis',
        diag_kind='kde'
    )
    plt.suptitle('Matriz de Dispersión: Evaluación de Separabilidad de Variables', y=1.02)
    plt.savefig(output_path)
    plt.close()
    print(f"[Imagen guardada]: {output_path}")

def calculate_vif(df: pd.DataFrame, target_col: str = 'quality') -> pd.DataFrame:
    """
    Calcula el VIF estandarizando temporalmente las variables independientes
    para diagnosticar la multicolinealidad.
    """
    X = df.drop(columns=[target_col])
    
    # Escalado únicamente para evaluar VIF
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled_const = add_constant(X_scaled)
    
    vif_data = pd.DataFrame({
        "Variable": X_scaled_const.columns,
        "VIF": [variance_inflation_factor(X_scaled_const.values, i) for i in range(X_scaled_const.shape[1])]
    })
    
    vif_filtered = vif_data[vif_data["Variable"] != "const"].sort_values(by="VIF", ascending=False)
    
    print("\n--- Resultados de VIF Real (Variables Estandarizadas) ---")
    print(vif_filtered.to_string(index=False))
    
    return vif_filtered


def create_target_variable(df: pd.DataFrame, threshold: int = 5)-> pd.DataFrame:
    df_transformed = df.copy()
    # Define la calidad binaria: 1 si es > threshold, 0 en otro caso
    df_transformed['target_quality'] = (df_transformed["quality"] > threshold).astype(int)
    print(f"Variable objetivo 'target_quality' creada usando umbral > {threshold}.")
    return df_transformed


if __name__ == "__main__":
    # Ruta del archivo de datos
    DATA_PATH = './data/winequality-white.csv'
    
    # 1. Carga y Limpieza
    df_raw = cargar_datos(DATA_PATH)
    df_clean = limpiar_datos(df_raw)
    
    # 2. Correlación
    plot_correlation_matrix(df_clean)
    
    # 3. Matriz de dispersión con características seleccionadas
    selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality']
    plot_pairplot(df_clean, selected_features)
    
    # 4. Diagnóstico de Multicolinealidad (VIF)
    vif_df = calculate_vif(df_clean, target_col='quality')