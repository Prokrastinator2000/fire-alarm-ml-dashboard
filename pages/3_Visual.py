import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Визуализации", page_icon="📈", layout="wide")

st.title("📈 Визуализации зависимостей в данных")

@st.cache_data
def load_data():
    return pd.read_csv("data/df_classification.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Файл data/df_classification.csv не найден!")
    st.stop()


st.header("1. Распределение целевой переменной (Fire Alarm)")

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))


if "Fire Alarm" in df.columns:
    counts = df["Fire Alarm"].value_counts()
    labels = ["Не сработал (0)", "Сработал (1)"]
    colors = ["#3498db", "#e74c3c"]
    axes1[0].pie(counts, labels=labels, autopct="%1.1f%%", colors=colors,
                 startangle=90, textprops={"fontsize": 12})
    axes1[0].set_title("Соотношение классов", fontsize=14)

    
    sns.countplot(x="Fire Alarm", data=df, ax=axes1[1], palette=colors)
    axes1[1].set_title("Количество по классам", fontsize=14)
    axes1[1].set_xlabel("Fire Alarm")
    axes1[1].set_ylabel("Количество")

    for p in axes1[1].patches:
        axes1[1].annotate(f'{int(p.get_height())}',
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='bottom', fontsize=12)

plt.tight_layout()
st.pyplot(fig1)


st.header("2. Корреляционная матрица")

numeric_cols = df.select_dtypes(include=[np.number]).columns
fig2, ax2 = plt.subplots(figsize=(14, 10))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax2, linewidths=0.5, annot_kws={"size": 8})
ax2.set_title("Корреляционная матрица признаков", fontsize=16)
plt.tight_layout()
st.pyplot(fig2)


st.header("3. Boxplot — распределение признаков по классам")

feature_cols = [c for c in numeric_cols if c != "Fire Alarm"]
selected_feature = st.selectbox("Выберите признак для boxplot:", feature_cols)

fig3, ax3 = plt.subplots(figsize=(10, 6))
if "Fire Alarm" in df.columns:
    sns.boxplot(x="Fire Alarm", y=selected_feature, data=df, ax=ax3,
                palette=["#3498db", "#e74c3c"])
    ax3.set_title(f"Распределение «{selected_feature}» по классам Fire Alarm",
                  fontsize=14)
    ax3.set_xlabel("Fire Alarm (0 = нет, 1 = да)")
    ax3.set_ylabel(selected_feature)
plt.tight_layout()
st.pyplot(fig3)


st.header("4. Гистограммы распределения признаков")

selected_features = st.multiselect(
    "Выберите признаки для гистограмм (макс. 6):",
    feature_cols,
    default=feature_cols[:4],
    max_selections=6
)

if selected_features:
    n_cols = 2
    n_rows = (len(selected_features) + 1) // 2
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes4 = axes4.flatten() if n_rows > 1 else [axes4] if len(selected_features) == 1 else axes4.flatten()

    for i, feat in enumerate(selected_features):
        if "Fire Alarm" in df.columns:
            df[df["Fire Alarm"] == 0][feat].hist(
                bins=40, alpha=0.5, label="No Alarm (0)",
                color="#3498db", ax=axes4[i]
            )
            df[df["Fire Alarm"] == 1][feat].hist(
                bins=40, alpha=0.5, label="Alarm (1)",
                color="#e74c3c", ax=axes4[i]
            )
        else:
            df[feat].hist(bins=40, ax=axes4[i], color="#3498db")
        axes4[i].set_title(feat, fontsize=12)
        axes4[i].legend()

    
    for j in range(len(selected_features), len(axes4)):
        axes4[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig4)
else:
    st.info("Выберите хотя бы один признак.")


st.header("5. Scatter plot — зависимость двух признаков")

col_a, col_b = st.columns(2)
with col_a:
    feat_x = st.selectbox("Признак по оси X:", feature_cols, index=0)
with col_b:
    feat_y = st.selectbox("Признак по оси Y:", feature_cols,
                          index=min(1, len(feature_cols) - 1))

fig5, ax5 = plt.subplots(figsize=(10, 6))
if "Fire Alarm" in df.columns:
    scatter = ax5.scatter(df[feat_x], df[feat_y], c=df["Fire Alarm"],
                          cmap="RdBu_r", alpha=0.4, s=10)
    plt.colorbar(scatter, ax=ax5, label="Fire Alarm")
else:
    ax5.scatter(df[feat_x], df[feat_y], alpha=0.4, s=10)

ax5.set_xlabel(feat_x)
ax5.set_ylabel(feat_y)
ax5.set_title(f"{feat_x} vs {feat_y}", fontsize=14)
plt.tight_layout()
st.pyplot(fig5)