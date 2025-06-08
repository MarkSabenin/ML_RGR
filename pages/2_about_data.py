import streamlit as st
from inference_utils import load_dataset

st.title("Информация о наборе данных")

df = load_dataset()

st.markdown("""
### Описание предметной области:
Набор данных содержит параметры околоземных объектов (астероидов), в том числе их диаметр, скорость, расстояние пролёта и абсолютную величину. Целевой признак — `hazardous`, указывающий на потенциальную опасность объекта.

### Признаки:
- **est_diameter_min, est_diameter_max** — минимальный и максимальный диаметр объекта (в км).
- **relative_velocity** — относительная скорость (в км/ч).
- **miss_distance** — расстояние пролета от Земли (в км).
- **absolute_magnitude** — абсолютная звездная величина.
- **hazardous** — является ли объект потенциально опасным; **целевой признак**.
- **mean_est_diameter** — среднее значение диаметра.
- **id_name** — уникальный идентификатор объекта (не используется в обучении).
""")

st.subheader("Пример данных:")
st.dataframe(df.head())

st.subheader("Форма и типы данных:")
import io

buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)
