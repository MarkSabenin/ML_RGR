import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from inference_utils import load_dataset

st.title("Визуализация данных")

df = load_dataset()

st.markdown("### Распределение признаков:")

columns = ['est_diameter_min', 'est_diameter_max', 'mean_est_diameter', 'absolute_magnitude', 'relative_velocity', 'miss_distance']
for column in columns:
    df.hist(column, bins=50, edgecolor='black')
    plt.yscale('log')
    st.pyplot(plt)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


