import pandas as pd


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    elif file_name.endswith(".json"):
        return pd.read_json(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file format. Please upload CSV, XLSX or JSON."
        )


# =========================================================
# DETECT COLUMNS
# =========================================================

def detect_columns(data):

    columns = list(data.columns)

    normalized = {
        str(column).lower().strip(): column
        for column in columns
    }

    title_names = [
        "title",
        "headline",
        "news_title",
        "heading",
        "news headline"
    ]

    text_names = [
        "text",
        "article",
        "content",
        "article_text",
        "article_content",
        "news_content",
        "body",
        "description"
    ]

    label_names = [
        "label",
        "class",
        "category",
        "target",
        "type",
        "result"
    ]

    title_column = None
    text_column = None
    label_column = None

    for name in title_names:
        if name in normalized:
            title_column = normalized[name]
            break

    for name in text_names:
        if name in normalized:
            text_column = normalized[name]
            break

    for name in label_names:
        if name in normalized:
            label_column = normalized[name]
            break

    return title_column, text_column, label_column


# =========================================================
# VALIDATE DATASET
# =========================================================

def validate_dataset(data):

    errors = []
    warnings = []

    title_column, text_column, label_column = detect_columns(data)

    if text_column is None:
        errors.append(
            "Article/Text column was not detected."
        )

    if label_column is None:
        errors.append(
            "Label column was not detected."
        )

    if len(data) == 0:
        errors.append("Dataset is empty.")

    if len(data) < 100:
        warnings.append(
            "Dataset contains fewer than 100 records."
        )

    if title_column is not None:

        missing_title = data[title_column].isnull().sum()

        if missing_title > 0:
            warnings.append(
                f"{missing_title} records have missing titles."
            )

    if text_column is not None:

        missing_text = data[text_column].isnull().sum()

        if missing_text > 0:
            warnings.append(
                f"{missing_text} records have missing article text."
            )

    label_counts = None

    if label_column is not None:

        label_counts = data[label_column].value_counts()

        if len(label_counts) < 2:
            errors.append(
                "Dataset must contain both Fake and Real classes."
            )

        if len(label_counts) > 2:
            warnings.append(
                "More than two label classes detected."
            )

    duplicate_count = int(
        data.duplicated().sum()
    )

    if duplicate_count > 0:
        warnings.append(
            f"{duplicate_count} duplicate rows detected."
        )

    balance_warning = False

    if label_counts is not None and len(label_counts) >= 2:

        largest = label_counts.max()
        smallest = label_counts.min()

        if smallest > 0:

            ratio = largest / smallest

            if ratio > 5:

                balance_warning = True

                warnings.append(
                    "Dataset is highly imbalanced."
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "title_column": title_column,
        "text_column": text_column,
        "label_column": label_column,
        "label_counts": label_counts,
        "duplicate_count": duplicate_count,
        "balance_warning": balance_warning
    }


# =========================================================
# STANDARDIZE DATASET
# =========================================================

def standardize_dataset(data):

    title_column, text_column, label_column = detect_columns(data)

    if text_column is None:
        raise ValueError(
            "Text/article column could not be detected."
        )

    if label_column is None:
        raise ValueError(
            "Label column could not be detected."
        )

    result = pd.DataFrame()

    # TITLE
    if title_column is not None:

        result["title"] = (
            data[title_column]
            .fillna("")
            .astype(str)
        )

    else:

        result["title"] = ""

    # ARTICLE TEXT
    result["text"] = (
        data[text_column]
        .fillna("")
        .astype(str)
    )

    # LABEL
    result["label"] = data[label_column]

    # -----------------------------------------------------
    # NORMALIZE LABELS
    # -----------------------------------------------------

    def normalize_label(value):

        value = str(value).strip().lower()

        if value in [
            "0",
            "fake",
            "false",
            "f",
            "fake news"
        ]:
            return 0

        if value in [
            "1",
            "real",
            "true",
            "r",
            "real news"
        ]:
            return 1

        return None

    result["label"] = (
        result["label"]
        .apply(normalize_label)
    )

    # Remove unknown labels
    result = result.dropna(
        subset=["label"]
    )

    result["label"] = (
        result["label"]
        .astype(int)
    )

    # -----------------------------------------------------
    # COMBINE TITLE + TEXT
    # -----------------------------------------------------

    result["content"] = (
        result["title"]
        + " "
        + result["text"]
    ).str.strip()

    # Remove empty content
    result = result[
        result["content"].str.len() > 0
    ]

    return result[
        ["content", "label"]
    ]