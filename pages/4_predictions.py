import streamlit as st
from inference_utils import load_models, preprocess_input_row, predict_all_models

st.title("Инференс моделей")

st.markdown("""
Введите параметры объекта, чтобы получить предсказания от всех моделей, обученных на задаче определения потенциально опасных околоземных объектов.
""")

# Поля ввода
col1, col2, col3 = st.columns(3)

with col1:
    est_diameter_min = st.number_input("Минимальный диаметр (км)", min_value=0.0, value=0.1)
    miss_distance = st.number_input("Расстояние пролета (км)", min_value=0.0, value=7500000.0)

with col2:
    est_diameter_max = st.number_input("Максимальный диаметр (км)", min_value=0.0, value=0.3)
    absolute_magnitude = st.number_input("Абсолютная величина", value=22.0)

with col3:
    relative_velocity = st.number_input("Относительная скорость (км/ч)", min_value=0.0, value=25000.0)

# Загрузка моделей
@st.cache_resource
def load_all():
    return load_models()

models = load_all()

# Инференс
if st.button("Сделать предсказание"):
    try:
        X_scaled = preprocess_input_row(est_diameter_min, est_diameter_max,
                                        relative_velocity, miss_distance,
                                        absolute_magnitude,
                                        scaler=models["scaler"])
        label_encoder = models["label_encoder"]
        new_models = {k: v for k, v in models.items() if k not in {"scaler", "label_encoder"}}
        preds = predict_all_models(new_models, X_scaled, label_encoder)
        
        st.subheader("Результаты предсказаний:")
        st.write(f"**KNN:** {'Опасен' if preds['knn']['prediction'][0] else 'Не опасен'}")
        st.write(f"**GBC:** {'Опасен' if preds['gbc']['prediction'][0] else 'Не опасен'}")
        st.write(f"**Bagging:** {'Опасен' if preds['bagg']['prediction'][0] else 'Не опасен'}")
        st.write(f"**Stacking:** {'Опасен' if preds['stac']['prediction'][0] else 'Не опасен'}")
        st.write(f"**LightGBM:** {'Опасен' if preds['lgbm']['prediction'][0] else 'Не опасен'}")
        st.write(f"**FCNN:** {'Опасен' if preds['fcnn']['prediction'][0] else 'Не опасен'}")

    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
