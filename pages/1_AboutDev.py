import streamlit as st

st.set_page_config(page_title="О разработчике", page_icon="👤", layout="wide")

st.title("👤 О разработчике")

col1, col2 = st.columns([1, 2])

with col1:
    try:
        st.image("photo.jpg", caption="Фото разработчика", width=250)
    except:
        st.info("📷 Фото не найдено. Поместите файл photo.jpg в корень проекта.")

with col2:
    st.markdown("""
    ### ФИО
    **Ильин Максим Викторович**

    ### Учебная группа
    **ФИТ-231**

    ### Тема РГР
    **«Разработка Web-приложения (дашборда) для инференса (вывода)
    моделей ML и анализа данных»**

    ### Дисциплина
    Машинное обучение

    ---

    ### Описание проекта
    В рамках данной расчётно-графической работы было разработано
    веб-приложение на основе библиотеки **Streamlit**, предназначенное
    для инференса шести моделей машинного обучения, обученных на
    датасете детектора дыма (Smoke Detection Dataset).

    Модели были обучены с подбором гиперпараметров через
    **GridSearchCV**, **RandomizedSearchCV** и **Optuna**, а затем
    сериализованы в формате `.joblib`.

    ---

    ### Контакты
    - 🔗 GitHub: [github.com/Prokrastinator2000](https://github.com/Prokrastinator2000)
    """)