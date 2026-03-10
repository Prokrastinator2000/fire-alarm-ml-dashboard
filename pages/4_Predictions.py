import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Предсказание", page_icon="🔮", layout="wide")

st.title("🔮 Предсказание пожарной тревоги (Fire Alarm)")


MODEL_DIR = "saved_models"

MODEL_FILES = {
    "DecisionTreeClassifier": "DecisionTreeClassifier.joblib",
    "GradientBoostingClassifier": "GradientBoostingClassifier.joblib",
    "LightGBMClassifier": "LightGBMClassifier.joblib",
    "BaggingClassifier": "BaggingClassifier.joblib",
    "StackingClassifier": "StackingClassifier.joblib",
    "XGBoostClassifier": "XGBoostClassifier.joblib",
}

@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f"⚠️ Модель {filename} не найдена в {MODEL_DIR}/")
    return models

models = load_models()

if not models:
    st.error("❌ Ни одна модель не загружена. Проверьте папку saved_models/")
    st.stop()


@st.cache_data
def load_data():
    return pd.read_csv("data/df_classification.csv")

try:
    df_full = load_data()
except:
    df_full = None


sample_model = list(models.values())[0]


if df_full is not None:
    all_feature_cols = [c for c in df_full.columns if c != "Fire Alarm"]
else:
    all_feature_cols = [
        "UTC", "Temperature[C]", "Humidity[%]", "TVOC[ppb]", "eCO2[ppm]",
        "Raw H2", "Raw Ethanol", "Pressure[hPa]", "PM1.0", "PM2.5",
        "NC0.5", "NC1.0", "NC2.5", "CNT"
    ]


st.markdown("---")
st.header("📥 Ввод данных")

input_method = st.radio(
    "Выберите способ ввода данных:",
    ["✏️ Ручной ввод", "📁 Загрузка CSV-файла"],
    horizontal=True
)


def clean_feature_names(df):
    """Приводит имена столбцов к формату, который использовался при обучении XGBoost"""
    df = df.copy()
    df.columns = [
        str(c)
        .replace("[", "_")
        .replace("]", "_")
        .replace("<", "lt_")
        .replace(">", "gt_")
        for c in df.columns
    ]
    return df


if input_method == "✏️ Ручной ввод":
    st.subheader("Введите показания датчиков:")

    
    feature_config = {}
    for col in all_feature_cols:
        if df_full is not None and col in df_full.columns:
            col_data = df_full[col]
            feature_config[col] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
            }
        else:
            feature_config[col] = {
                "min": 0.0, "max": 100.0, "mean": 50.0, "median": 50.0
            }

    
    input_values = {}
    cols = st.columns(3)

    for i, col_name in enumerate(all_feature_cols):
        cfg = feature_config[col_name]
        with cols[i % 3]:
            input_values[col_name] = st.number_input(
                f"**{col_name}**",
                min_value=cfg["min"] - abs(cfg["min"]) * 0.5,
                max_value=cfg["max"] + abs(cfg["max"]) * 0.5,
                value=cfg["median"],
                step=(cfg["max"] - cfg["min"]) / 100,
                format="%.4f",
                help=f"Диапазон в датасете: [{cfg['min']:.2f}, {cfg['max']:.2f}], "
                     f"среднее: {cfg['mean']:.2f}"
            )

    
    st.markdown("---")
    predict_button = st.button("🔮 Предсказать", type="primary", use_container_width=True)

    if predict_button:
        input_df = pd.DataFrame([input_values])

        st.subheader("📋 Введённые данные:")
        st.dataframe(input_df, use_container_width=True)

        st.subheader("📊 Результаты предсказания:")

        results_cols = st.columns(3)

        for i, (model_name, model) in enumerate(models.items()):
            with results_cols[i % 3]:
                try:
                    
                    if "XGBoost" in model_name:
                        pred_df = clean_feature_names(input_df)
                    else:
                        pred_df = input_df

                    prediction = model.predict(pred_df)[0]

                    
                    proba = None
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(pred_df)[0]
                        except:
                            pass

                    
                    if prediction == 1:
                        st.error(f"🔴 **{model_name}**")
                        st.markdown("### 🚨 ТРЕВОГА!")
                        st.markdown("**Fire Alarm = 1** — Обнаружен дым!")
                    else:
                        st.success(f"🟢 **{model_name}**")
                        st.markdown("### ✅ Всё в порядке")
                        st.markdown("**Fire Alarm = 0** — Дым не обнаружен")

                    if proba is not None:
                        st.markdown(f"**Вероятность тревоги:** {proba[1]:.2%}")
                        st.progress(float(proba[1]))

                except Exception as e:
                    st.warning(f"⚠️ {model_name}: ошибка — {e}")





elif input_method == "📁 Загрузка CSV-файла":
    st.subheader("Загрузите CSV-файл с данными датчиков")

    st.markdown(f"""
    **Формат CSV:** файл должен содержать следующие столбцы:

    `{', '.join(all_feature_cols)}`
    """)

    uploaded_file = st.file_uploader("Выберите CSV-файл:", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            st.subheader("📋 Загруженные данные:")
            st.dataframe(input_df.head(20), use_container_width=True)
            st.info(f"Загружено строк: {len(input_df)}")

            
            if "Fire Alarm" in input_df.columns:
                y_true = input_df["Fire Alarm"]
                input_df = input_df.drop(columns=["Fire Alarm"])
            else:
                y_true = None

            
            input_features = input_df[[c for c in all_feature_cols if c in input_df.columns]]

            st.markdown("---")

            
            selected_model_name = st.selectbox(
                "Выберите модель:",
                list(models.keys()),
                key="csv_model_select"
            )

            
            predict_csv_button = st.button(
                "🔮 Предсказать для всех строк",
                type="primary",
                use_container_width=True,
                key="csv_predict_btn"
            )

            
            if predict_csv_button:
                st.session_state["csv_predictions_ready"] = True
                st.session_state["csv_selected_model"] = selected_model_name

            
            if "csv_predictions_ready" in st.session_state and st.session_state["csv_predictions_ready"]:
                current_model_name = selected_model_name
                model = models[current_model_name]

                st.subheader(f"📊 Результаты предсказания — {current_model_name}")

                try:
                    if "XGBoost" in current_model_name:
                        pred_features = clean_feature_names(input_features)
                    else:
                        pred_features = input_features

                    predictions = model.predict(pred_features)

                    result_df = input_features.copy()
                    result_df["Предсказание"] = predictions
                    result_df["Результат"] = result_df["Предсказание"].map(
                        {0: "✅ Нет тревоги", 1: "🚨 ТРЕВОГА!"}
                    )

                    if hasattr(model, "predict_proba"):
                        try:
                            probas = model.predict_proba(pred_features)
                            result_df["Вероятность тревоги"] = probas[:, 1]
                        except:
                            pass

                    if y_true is not None:
                        result_df["Истинное значение"] = y_true.values
                        from sklearn.metrics import accuracy_score, f1_score
                        acc = accuracy_score(y_true, predictions)
                        f1 = f1_score(y_true, predictions)
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Accuracy", f"{acc:.4f}")
                        col_m2.metric("F1-score", f"{f1:.4f}")
                        col_m3.metric("Всего строк", len(predictions))

                    st.dataframe(result_df, use_container_width=True)

                    
                    n_alarm = int((predictions == 1).sum())
                    n_safe = int((predictions == 0).sum())

                    col1, col2 = st.columns(2)
                    col1.metric("🚨 Тревога (1)", n_alarm)
                    col2.metric("✅ Нет тревоги (0)", n_safe)

                   
                    csv_result = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Скачать результаты (CSV)",
                        data=csv_result,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"❌ Ошибка предсказания: {e}")

        except Exception as e:
            st.error(f"❌ Ошибка чтения файла: {e}")


st.markdown("---")
st.header("ℹ️ Информация о загруженных моделях")

for name, model in models.items():
    with st.expander(f"🤖 {name}"):
        st.code(str(model), language="text")
        if hasattr(model, "get_params"):
            try:
                params = model.get_params()
                st.json(params)
            except:
                st.info("Параметры недоступны")