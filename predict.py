import pandas as pd
import joblib

def predict_wine_quality(new_data: pd.DataFrame, model_path: str = 'best_wine_model.pkl'):
    # 1. Cargar el pipeline guardado
    model = joblib.load(model_path)

    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:,1]

    return predictions, probabilities


if __name__ == "__main__":
    # Ejemplo: Datos químicos de un nuevo vino (sin la columna 'density' ni 'quality')
    sample_wine = pd.DataFrame([{
        'fixed acidity': 7.0,
        'volatile acidity': 0.27,
        'citric acid': 0.36,
        'residual sugar': 2.07,
        'chlorides': 0.045,
        'free sulfur dioxide': 45.0,
        'total sulfur dioxide': 170.0,
        'pH': 3.00,
        'sulphates': 0.45,
        'alcohol': 12.8
    }])

    pred, proba = predict_wine_quality(sample_wine)
    label = "Alta Calidad (>5)" if pred[0] == 1 else "Calidad Media/Baja (<=5)"
    print(f"Predicción para el nuevo vino: {label}")
    print(f"Probabilidad de ser Alta Calidad: {proba[0]:.2%}")