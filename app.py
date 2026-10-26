import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime

from dataset_processor import (
    load_dataset,
    detect_columns,
    validate_dataset,
    standardize_dataset
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# BLACK + GOLD PROFESSIONAL THEME
# =========================================================

st.markdown("""
<style>

/* ================================
   MAIN BACKGROUND
================================ */

.stApp {
    background:
        radial-gradient(
            circle at 85% 10%,
            rgba(212, 175, 55, 0.08),
            transparent 25%
        ),
        #080808;

    color: #F5F5F5;
}


/* ================================
   MAIN CONTAINER
================================ */

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================================
   STREAMLIT HEADER
================================ */

[data-testid="stHeader"] {
    background: #080808;
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #050505,
            #0B0B0B
        );

    border-right: 1px solid #3A2E12;
}

[data-testid="stSidebar"] * {
    color: #F5F5F5;
}


/* Sidebar navigation */

[data-testid="stSidebar"] .stRadio label {
    padding: 8px 5px;
}


/* ================================
   INPUT BOXES
================================ */

.stTextInput input,
.stTextArea textarea {

    background-color: #111111 !important;

    color: #FFFFFF !important;

    border: 1px solid #383838 !important;

    border-radius: 10px !important;
}


.stTextInput input:focus,
.stTextArea textarea:focus {

    border: 1px solid #D4AF37 !important;

    box-shadow: 0 0 0 1px #D4AF37 !important;
}


/* Input labels */

.stTextInput label,
.stTextArea label,
.stFileUploader label {

    color: #E5E5E5 !important;

    font-weight: 600 !important;
}


/* ================================
   BUTTONS
================================ */

.stButton > button {

    background:
        linear-gradient(
            135deg,
            #D4AF37,
            #B8860B
        ) !important;

    color: #080808 !important;

    border: none !important;

    border-radius: 10px !important;

    min-height: 46px !important;

    font-weight: 800 !important;

    transition: 0.2s;
}


.stButton > button:hover {

    background:
        linear-gradient(
            135deg,
            #E8C95A,
            #D4AF37
        ) !important;

    color: #000000 !important;

    transform: translateY(-1px);
}


/* ================================
   METRIC CARDS
================================ */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            #151515,
            #0E0E0E
        );

    border: 1px solid #443612;

    border-radius: 15px;

    padding: 18px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.35);
}


[data-testid="stMetricLabel"] {

    color: #A3A3A3 !important;
}


[data-testid="stMetricValue"] {

    color: #D4AF37 !important;

    font-weight: 800 !important;
}


/* ================================
   FILE UPLOADER
================================ */

[data-testid="stFileUploader"] {

    background: #111111;

    border: 1px dashed #66531C;

    border-radius: 12px;
}


/* ================================
   DATAFRAME
================================ */

[data-testid="stDataFrame"] {

    border: 1px solid #443612;

    border-radius: 10px;

    overflow: hidden;
}


/* ================================
   HEADINGS
================================ */

h1,
h2,
h3,
h4 {

    color: #FFFFFF !important;
}


/* ================================
   GOLD TEXT
================================ */

.gold-text {

    color: #D4AF37 !important;
}


/* ================================
   DIVIDER
================================ */

hr {

    border-color: #3A2E12 !important;
}


/* ================================
   ALERTS
================================ */

[data-testid="stAlert"] {

    border-radius: 12px;
}


/* ================================
   CAPTION
================================ */

.stCaption {

    color: #A3A3A3 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LANGUAGE SYSTEM
# =========================================================

if "language" not in st.session_state:
    st.session_state.language = "English"


TRANSLATIONS = {

    "English": {

        "caption":
            "Intelligent News Verification",

        "navigation":
            "NAVIGATION",

        "analyze":
            "Analyze News",

        "dataset":
            "Dataset Hub",

        "model_info":
            "Model Information",

        "history":
            "Prediction History",

        "model_online":
            "● AI Model Online",

        "model_offline":
            "● Model Offline",

        "main_caption":
            "Intelligent Fake News Detection Platform powered by Machine Learning",

        "model_type":
            "● Logistic Regression + TF-IDF",

        "analyze_title":
            "Analyze News",

        "analyze_caption":
            "Enter a headline or article to analyze its authenticity.",

        "accuracy":
            "🎯 Model Accuracy",

        "training_records":
            "📚 Training Records",

        "predictions":
            "🔎 Predictions Made",

        "headline":
            "News Headline",

        "headline_placeholder":
            "Enter the news headline...",

        "article":
            "Article Content",

        "article_placeholder":
            "Paste the complete news article here...",

        "analyze_button":
            "🔎 Analyze News",

        "sample_news":
            "🧪 Try Sample News",

        "input_warning":
            "Please enter a headline or article first.",

        "fake":
            "🔴 Likely Fake News",

        "fake_text":
            "The model predicts this content as Fake.",

        "real":
            "🟢 Likely Real News",

        "real_text":
            "The model predicts this content as Real.",

        "confidence":
            "Confidence",

        "important":
            "Important: This is an ML-based prediction, "
            "not a guarantee of factual truth. "
            "Always verify important information using "
            "reliable sources.",

        "dataset_caption":
            "Upload and validate your own labelled news dataset.",

        "upload_dataset":
            "Upload Dataset",

        "supported":
            "Supported formats: CSV, Excel and JSON",

        "loaded":
            "Dataset loaded successfully.",

        "records":
            "Records",

        "columns":
            "Columns",

        "duplicates":
            "Duplicates",

        "missing":
            "Missing Values",

        "column_detection":
            "Automatic Column Detection",

        "title":
            "Title",

        "text":
            "Text",

        "label":
            "Label",

        "title_not":
            "Title not detected",

        "text_not":
            "Text not detected",

        "label_not":
            "Label not detected",

        "validation":
            "Dataset Validation",

        "validation_success":
            "Dataset passed basic validation.",

        "validation_failed":
            "Dataset validation failed.",

        "preview":
            "Dataset Preview",

        "prepare":
            "🔄 Prepare Dataset",

        "prepared":
            "Dataset prepared successfully.",

        "usable":
            "Usable Records",

        "fake_count":
            "Fake",

        "real_count":
            "Real",

        "processing_failed":
            "Dataset processing failed.",

        "read_failed":
            "Could not read this dataset.",

        "model_title":
            "Model Information",

        "model_caption":
            "Technical overview of the machine learning system.",

        "machine_model":
            "🤖 Machine Learning Model",

        "algorithm":
            "Algorithm",

        "feature":
            "Feature Extraction",

        "testing_records":
            "Testing Records",

        "performance":
            "📊 Model Performance",

        "class":
            "Class",

        "precision":
            "Precision",

        "recall":
            "Recall",

        "f1":
            "F1 Score",

        "pipeline":
            "⚙️ Machine Learning Pipeline",

        "history_title":
            "Prediction History",

        "history_caption":
            "Review predictions made during the current session.",

        "no_history":
            "No predictions have been made yet.",

        "clear_history":
            "🗑️ Clear Prediction History",

        "download_history":
            "📥 Download Prediction History",

        "download_ready":
            "Prediction history is ready to download.",

        "footer":
            "TruthLens AI • Machine Learning Based News Analysis",

        "footer2":
            "Built with Python • Scikit-learn • Streamlit",

        "model_error":
            "Unable to load the machine learning model."

    },


    "हिंदी": {

        "caption":
            "इंटेलिजेंट न्यूज़ वेरिफिकेशन",

        "navigation":
            "नेविगेशन",

        "analyze":
            "न्यूज़ का विश्लेषण",

        "dataset":
            "डेटासेट हब",

        "model_info":
            "मॉडल की जानकारी",

        "history":
            "प्रेडिक्शन हिस्ट्री",

        "model_online":
            "● AI मॉडल ऑनलाइन",

        "model_offline":
            "● AI मॉडल ऑफलाइन",

        "main_caption":
            "Machine Learning द्वारा संचालित इंटेलिजेंट फेक न्यूज़ डिटेक्शन प्लेटफॉर्म",

        "model_type":
            "● Logistic Regression + TF-IDF",

        "analyze_title":
            "न्यूज़ का विश्लेषण करें",

        "analyze_caption":
            "न्यूज़ की सत्यता जांचने के लिए हेडलाइन या आर्टिकल दर्ज करें।",

        "accuracy":
            "🎯 मॉडल सटीकता",

        "training_records":
            "📚 ट्रेनिंग रिकॉर्ड्स",

        "predictions":
            "🔎 किए गए प्रेडिक्शन",

        "headline":
            "न्यूज़ हेडलाइन",

        "headline_placeholder":
            "न्यूज़ हेडलाइन दर्ज करें...",

        "article":
            "आर्टिकल कंटेंट",

        "article_placeholder":
            "पूरा न्यूज़ आर्टिकल यहाँ पेस्ट करें...",

        "analyze_button":
            "🔎 न्यूज़ का विश्लेषण करें",

        "sample_news":
            "🧪 सैंपल न्यूज़ आज़माएँ",

        "input_warning":
            "कृपया पहले हेडलाइन या आर्टिकल दर्ज करें।",

        "fake":
            "🔴 संभावित फेक न्यूज़",

        "fake_text":
            "मॉडल के अनुसार यह कंटेंट फेक न्यूज़ है।",

        "real":
            "🟢 संभावित रियल न्यूज़",

        "real_text":
            "मॉडल के अनुसार यह कंटेंट रियल न्यूज़ है।",

        "confidence":
            "विश्वसनीयता",

        "important":
            "महत्वपूर्ण: यह Machine Learning आधारित अनुमान है, "
            "सत्यता की गारंटी नहीं। महत्वपूर्ण जानकारी को "
            "विश्वसनीय स्रोतों से हमेशा सत्यापित करें।",

        "dataset_caption":
            "अपना लेबल किया हुआ न्यूज़ डेटासेट अपलोड और वैलिडेट करें।",

        "upload_dataset":
            "डेटासेट अपलोड करें",

        "supported":
            "समर्थित फॉर्मेट: CSV, Excel और JSON",

        "loaded":
            "डेटासेट सफलतापूर्वक लोड हो गया।",

        "records":
            "रिकॉर्ड्स",

        "columns":
            "कॉलम",

        "duplicates":
            "डुप्लीकेट्स",

        "missing":
            "मिसिंग वैल्यूज़",

        "column_detection":
            "ऑटोमैटिक कॉलम डिटेक्शन",

        "title":
            "टाइटल",

        "text":
            "टेक्स्ट",

        "label":
            "लेबल",

        "title_not":
            "टाइटल नहीं मिला",

        "text_not":
            "टेक्स्ट नहीं मिला",

        "label_not":
            "लेबल नहीं मिला",

        "validation":
            "डेटासेट वैलिडेशन",

        "validation_success":
            "डेटासेट ने बेसिक वैलिडेशन पास कर लिया।",

        "validation_failed":
            "डेटासेट वैलिडेशन फेल हो गया।",

        "preview":
            "डेटासेट प्रीव्यू",

        "prepare":
            "🔄 डेटासेट तैयार करें",

        "prepared":
            "डेटासेट सफलतापूर्वक तैयार हो गया।",

        "usable":
            "उपयोग योग्य रिकॉर्ड्स",

        "fake_count":
            "फेक",

        "real_count":
            "रियल",

        "processing_failed":
            "डेटासेट प्रोसेस करने में समस्या आई।",

        "read_failed":
            "यह डेटासेट पढ़ा नहीं जा सका।",

        "model_title":
            "मॉडल की जानकारी",

        "model_caption":
            "Machine Learning सिस्टम की तकनीकी जानकारी।",

        "machine_model":
            "🤖 Machine Learning मॉडल",

        "algorithm":
            "एल्गोरिदम",

        "feature":
            "फीचर एक्सट्रैक्शन",

        "testing_records":
            "टेस्टिंग रिकॉर्ड्स",

        "performance":
            "📊 मॉडल परफॉर्मेंस",

        "class":
            "क्लास",

        "precision":
            "प्रिसीजन",

        "recall":
            "रिकॉल",

        "f1":
            "F1 स्कोर",

        "pipeline":
            "⚙️ Machine Learning पाइपलाइन",

        "history_title":
            "प्रेडिक्शन हिस्ट्री",

        "history_caption":
            "इस सेशन में किए गए प्रेडिक्शन देखें।",

        "no_history":
            "अभी तक कोई प्रेडिक्शन नहीं किया गया है।",

        "clear_history":
            "🗑️ प्रेडिक्शन हिस्ट्री साफ करें",

        "download_history":
            "📥 प्रेडिक्शन हिस्ट्री डाउनलोड करें",

        "download_ready":
            "प्रेडिक्शन हिस्ट्री डाउनलोड के लिए तैयार है।",

        "footer":
            "TruthLens AI • Machine Learning आधारित न्यूज़ विश्लेषण",

        "footer2":
            "Python • Scikit-learn • Streamlit से बनाया गया",

        "model_error":
            "Machine Learning मॉडल लोड नहीं हो पाया।"
    }
}


# =========================================================
# LANGUAGE SELECTOR
# =========================================================

with st.sidebar:

    st.title("📰 TruthLens AI")

    st.caption(
        TRANSLATIONS[
            st.session_state.language
        ]["caption"]
    )

    st.divider()

    st.session_state.language = st.selectbox(
        "🌐 Language / भाषा",
        ["English", "हिंदी"],
        index=(
            0
            if st.session_state.language == "English"
            else 1
        )
    )

    lang = st.session_state.language

    T = TRANSLATIONS[lang]


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    base_dir = Path(__file__).resolve().parent

    model_dir = base_dir / "model"

    model_path = (
        model_dir /
        "fake_news_model.pkl"
    )

    vectorizer_path = (
        model_dir /
        "tfidf_vectorizer.pkl"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not vectorizer_path.exists():

        raise FileNotFoundError(
            f"Vectorizer file not found: {vectorizer_path}"
        )

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(file)

    with open(
        vectorizer_path,
        "rb"
    ) as file:

        vectorizer = pickle.load(file)

    return model, vectorizer


# =========================================================
# MODEL INITIALIZATION
# =========================================================

try:

    model, vectorizer = load_model()

    model_status = True

except Exception as error:

    model_status = False

    model_error = str(error)


# =========================================================
# SESSION STATE
# =========================================================

if "history" not in st.session_state:

    st.session_state.history = []


# Sample news state

if "sample_title" not in st.session_state:

    st.session_state.sample_title = ""


if "sample_article" not in st.session_state:

    st.session_state.sample_article = ""


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.divider()

    page = st.radio(
        T["navigation"],
        [
            T["analyze"],
            T["dataset"],
            T["model_info"],
            T["history"]
        ]
    )

    st.divider()

    if model_status:

        st.success(
            T["model_online"]
        )

    else:

        st.error(
            T["model_offline"]
        )


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "📰 TruthLens AI"
)

st.caption(
    T["main_caption"]
)

st.markdown(
    T["model_type"]
)

st.divider()


# =========================================================
# MODEL ERROR
# =========================================================

if not model_status:

    st.error(
        T["model_error"]
    )

    st.code(
        model_error
    )

    st.stop()


# =========================================================
# ANALYZE NEWS
# =========================================================

if page == T["analyze"]:

    st.header(
        T["analyze_title"]
    )

    st.caption(
        T["analyze_caption"]
    )


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            T["accuracy"],
            "98.64%"
        )


    with col2:

        st.metric(
            T["training_records"],
            "44,898"
        )


    with col3:

        st.metric(
            T["predictions"],
            len(
                st.session_state.history
            )
        )


    st.write("")


    # -----------------------------------------------------
    # TRY SAMPLE NEWS
    # -----------------------------------------------------

    if st.button(
        T["sample_news"],
        use_container_width=True
    ):

        if lang == "English":

            st.session_state.sample_title = (
                "Government Announces New Digital Education Initiative"
            )

            st.session_state.sample_article = (
                "The government has announced a new initiative "
                "to improve digital education opportunities for "
                "students across the country. The program will "
                "focus on improving digital infrastructure, "
                "online learning resources and teacher training."
            )

        else:

            st.session_state.sample_title = (
                "सरकार ने डिजिटल शिक्षा को बढ़ावा देने के लिए नई पहल की घोषणा की"
            )

            st.session_state.sample_article = (
                "सरकार ने देशभर के छात्रों के लिए डिजिटल शिक्षा "
                "के अवसरों को बेहतर बनाने के लिए एक नई पहल की घोषणा की है। "
                "इस कार्यक्रम का उद्देश्य डिजिटल इंफ्रास्ट्रक्चर, "
                "ऑनलाइन शिक्षण संसाधनों और शिक्षकों के प्रशिक्षण में सुधार करना है।"
            )

        st.rerun()


    # -----------------------------------------------------
    # NEWS INPUT
    # -----------------------------------------------------

    title = st.text_input(

        T["headline"],

        value=st.session_state.sample_title,

        placeholder=
        T["headline_placeholder"]
    )


    article = st.text_area(

        T["article"],

        value=st.session_state.sample_article,

        placeholder=
        T["article_placeholder"],

        height=260
    )


    st.write("")


    analyze = st.button(

        T["analyze_button"],

        use_container_width=True
    )


    # -----------------------------------------------------
    # ANALYZE
    # -----------------------------------------------------

    if analyze:

        if (
            not title.strip()
            and
            not article.strip()
        ):

            st.warning(
                T["input_warning"]
            )

        else:

            content = (
                title
                + " "
                + article
            ).strip()


            features = (
                vectorizer.transform(
                    [content]
                )
            )


            prediction = (
                model.predict(
                    features
                )[0]
            )


            probabilities = (
                model.predict_proba(
                    features
                )[0]
            )


            confidence = (
                max(probabilities)
                * 100
            )


            current_time = (
                datetime.now()
                .strftime(
                    "%d %b %Y, %I:%M %p"
                )
            )


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if prediction == 0:

                st.error(
                    T["fake"]
                )

                st.write(
                    T["fake_text"]
                )

                st.write(
                    f"{T['confidence']}: "
                    f"{confidence:.2f}%"
                )

                result = "Fake"


            else:

                st.success(
                    T["real"]
                )

                st.write(
                    T["real_text"]
                )

                st.write(
                    f"{T['confidence']}: "
                    f"{confidence:.2f}%"
                )

                result = "Real"


            # ------------------------------------------------
            # SAVE PREDICTION
            # ------------------------------------------------

            st.session_state.history.append(

                {
                    "Time":
                        current_time,

                    "Result":
                        result,

                    "Confidence":
                        f"{confidence:.2f}%",

                    "Headline":
                        title[:100]
                }
            )


            st.info(
                T["important"]
            )


# =========================================================
# DATASET HUB
# =========================================================

elif page == T["dataset"]:

    st.header(
        T["dataset"]
    )

    st.caption(
        T["dataset_caption"]
    )


    uploaded_file = st.file_uploader(

        T["upload_dataset"],

        type=[
            "csv",
            "xlsx",
            "json"
        ],

        help=T["supported"]
    )


    if uploaded_file:

        try:

            data = load_dataset(
                uploaded_file
            )


            st.success(
                T["loaded"]
            )


            # ------------------------------------------------
            # DATASET METRICS
            # ------------------------------------------------

            c1, c2, c3, c4 = (
                st.columns(4)
            )


            with c1:

                st.metric(
                    T["records"],
                    len(data)
                )


            with c2:

                st.metric(
                    T["columns"],
                    len(data.columns)
                )


            with c3:

                st.metric(
                    T["duplicates"],
                    int(
                        data.duplicated()
                        .sum()
                    )
                )


            with c4:

                st.metric(
                    T["missing"],
                    int(
                        data.isnull()
                        .sum()
                        .sum()
                    )
                )


            st.divider()


            # ------------------------------------------------
            # COLUMN DETECTION
            # ------------------------------------------------

            st.subheader(
                T["column_detection"]
            )


            title_col, text_col, label_col = (
                detect_columns(data)
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                if title_col:

                    st.success(
                        f"{T['title']}: "
                        f"{title_col}"
                    )

                else:

                    st.warning(
                        T["title_not"]
                    )


            with c2:

                if text_col:

                    st.success(
                        f"{T['text']}: "
                        f"{text_col}"
                    )

                else:

                    st.error(
                        T["text_not"]
                    )


            with c3:

                if label_col:

                    st.success(
                        f"{T['label']}: "
                        f"{label_col}"
                    )

                else:

                    st.error(
                        T["label_not"]
                    )


            st.divider()


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            st.subheader(
                T["validation"]
            )


            validation = (
                validate_dataset(data)
            )


            if validation["valid"]:

                st.success(
                    T["validation_success"]
                )

            else:

                st.error(
                    T["validation_failed"]
                )

                for error in (
                    validation["errors"]
                ):

                    st.error(
                        error
                    )


            if validation["warnings"]:

                for warning in (
                    validation["warnings"]
                ):

                    st.warning(
                        warning
                    )


            st.divider()


            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            st.subheader(
                T["preview"]
            )


            st.dataframe(

                data.head(10),

                use_container_width=True,

                hide_index=True
            )


            # ------------------------------------------------
            # PREPARE DATASET
            # ------------------------------------------------

            if validation["valid"]:

                if st.button(

                    T["prepare"],

                    use_container_width=True
                ):

                    try:

                        standardized = (
                            standardize_dataset(
                                data
                            )
                        )


                        st.success(
                            T["prepared"]
                        )


                        fake_count = int(

                            (
                                standardized[
                                    "label"
                                ] == 0
                            ).sum()
                        )


                        real_count = int(

                            (
                                standardized[
                                    "label"
                                ] == 1
                            ).sum()
                        )


                        c1, c2, c3 = (
                            st.columns(3)
                        )


                        with c1:

                            st.metric(
                                T["usable"],
                                len(
                                    standardized
                                )
                            )


                        with c2:

                            st.metric(
                                T["fake_count"],
                                fake_count
                            )


                        with c3:

                            st.metric(
                                T["real_count"],
                                real_count
                            )


                    except Exception as error:

                        st.error(
                            T[
                                "processing_failed"
                            ]
                        )

                        st.code(
                            str(error)
                        )


        except Exception as error:

            st.error(
                T["read_failed"]
            )

            st.code(
                str(error)
            )


# =========================================================
# MODEL INFORMATION
# =========================================================

elif page == T["model_info"]:

    st.header(
        T["model_title"]
    )

    st.caption(
        T["model_caption"]
    )


    st.subheader(
        T["machine_model"]
    )


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            T["algorithm"],
            "Logistic Regression"
        )


    with c2:

        st.metric(
            T["feature"],
            "TF-IDF"
        )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            T["training_records"],
            "35,918"
        )


    with c2:

        st.metric(
            T["testing_records"],
            "8,980"
        )


    with c3:

        st.metric(
            T["accuracy"],
            "98.64%"
        )


    st.divider()


    st.subheader(
        T["performance"]
    )


    performance_df = pd.DataFrame(

        {

            T["class"]:
                [
                    "Fake",
                    "Real"
                ],

            T["precision"]:
                [
                    0.99,
                    0.98
                ],

            T["recall"]:
                [
                    0.98,
                    0.99
                ],

            T["f1"]:
                [
                    0.99,
                    0.99
                ]
        }
    )


    st.dataframe(

        performance_df,

        use_container_width=True,

        hide_index=True
    )


    st.divider()


    st.subheader(
        T["pipeline"]
    )


    st.code(
        """
News Input
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Fake / Real
"""
    )


# =========================================================
# PREDICTION HISTORY
# =========================================================

elif page == T["history"]:

    st.header(
        T["history_title"]
    )

    st.caption(
        T["history_caption"]
    )


    if not st.session_state.history:

        st.info(
            T["no_history"]
        )


    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )


        st.dataframe(

            history_df,

            use_container_width=True,

            hide_index=True
        )


        st.write("")


        # ------------------------------------------------
        # DOWNLOAD HISTORY
        # ------------------------------------------------

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(

            T["download_history"],

            data=csv_data,

            file_name=
            "truthlens_prediction_history.csv",

            mime="text/csv",

            use_container_width=True
        )


        st.write("")


        # ------------------------------------------------
        # CLEAR HISTORY
        # ------------------------------------------------

        if st.button(
            T["clear_history"]
        ):

            st.session_state.history = []

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    T["footer"]
)

st.caption(
    T["footer2"]
)