import streamlit as st
import pandas as pd

st.set_page_config(page_title="О датасете", page_icon="📊", layout="wide")

st.title("📊 Информация о наборе данных")


@st.cache_data
def load_data():
    return pd.read_csv("data/df_classification.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Файл data/df_classification.csv не найден!")
    st.stop()


st.header("1. Предметная область")
st.markdown("""
Датасет **Smoke Detection Dataset** содержит показания различных датчиков,
установленных в помещении, для определения наличия задымления.

**Целевая переменная:** `Fire Alarm`
- `0` — пожарная сигнализация **НЕ сработала** (дым не обнаружен)
- `1` — пожарная сигнализация **СРАБОТАЛА** (обнаружен дым)

Датасет используется для обучения моделей бинарной классификации,
которые по показаниям датчиков определяют, нужно ли поднимать тревогу.
""")


st.header("2. Описание признаков")

feature_descriptions = {
    "UTC": "Временная метка (Unix timestamp)",
    "Temperature[C]": "Температура воздуха, °C",
    "Humidity[%]": "Относительная влажность воздуха, %",
    "TVOC[ppb]": "Общее содержание летучих органических соединений, ppb",
    "eCO2[ppm]": "Эквивалентная концентрация CO₂, ppm",
    "Raw H2": "Сырое значение датчика водорода",
    "Raw Ethanol": "Сырое значение датчика этанола",
    "Pressure[hPa]": "Атмосферное давление, гПа",
    "PM1.0": "Концентрация частиц PM1.0, мкг/м³",
    "PM2.5": "Концентрация частиц PM2.5, мкг/м³",
    "NC0.5": "Количество частиц размером ≥0.5 мкм",
    "NC1.0": "Количество частиц размером ≥1.0 мкм",
    "NC2.5": "Количество частиц размером ≥2.5 мкм",
    "CNT": "Счётчик сэмплов",
    "Fire Alarm": "Целевая переменная (0 или 1)"
}

desc_df = pd.DataFrame(
    list(feature_descriptions.items()),
    columns=["Признак", "Описание"]
)
st.table(desc_df)


st.header("3. Основные характеристики датасета")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📐 Строк", df.shape[0])
col2.metric("📏 Столбцов", df.shape[1])
col3.metric("❌ Пропуски", df.isnull().sum().sum())

if "Fire Alarm" in df.columns:
    balance = df["Fire Alarm"].value_counts()
    col4.metric("⚖️ Баланс (1/0)", f"{balance.get(1, 0)} / {balance.get(0, 0)}")


st.header("4. Первые строки датасета")
st.dataframe(df.head(20), use_container_width=True)


st.header("5. Статистическое описание")
st.dataframe(df.describe().T, use_container_width=True)


st.header("6. Типы данных")
dtypes_df = pd.DataFrame({
    "Признак": df.columns,
    "Тип": df.dtypes.astype(str).values,
    "Пропуски": df.isnull().sum().values,
    "Уникальных": df.nunique().values
})
st.dataframe(dtypes_df, use_container_width=True)


st.header("7. Особенности предобработки данных")
st.markdown("""
1. **Проверка пропусков** — пропущенные значения отсутствуют.
2. **Балансировка классов** — применён метод **SMOTE** (Synthetic Minority
   Over-sampling Technique) для устранения дисбаланса целевой переменной.
3. **Масштабирование** — признаки не масштабировались, так как
   используются модели на основе деревьев решений, которые инвариантны
   к масштабу признаков.
4. **Удаление нерелевантных признаков** — при необходимости удалены
   столбцы, не несущие информации для классификации (например, UTC).
""")