# Product Length Prediction Challenge (ACCEVENT - NIT JSR)

An end-to-end Machine Learning pipeline built in Python to predict product dimensions (`PRODUCT_LENGTH`) from e-commerce metadata (Titles, Bullet Points, and Product Type IDs) using [LightGBM](https://lightgbm.readthedocs.io/) and advanced feature engineering.

## Project Overview
Predicting physical dimensions from unstructured text descriptions (like titles and bullet points) presents a significant sparse-data and NLP challenge. This pipeline cleans raw tabular data, extracts physical dimensions/volumes via custom Regular Expressions, combines text features using TF-IDF vectorization, and trains gradient-boosted regression models.

## Key Features & Pipeline Steps
1. **Data Preprocessing & Cleaning:** Handled missing values across text attributes and filtered/clipped outlier lengths using percentile bounds.
2. **Feature Engineering:**
   - **Regex Extraction:** Parsed units (mm, cm, m, inches) directly from product titles to capture explicit lengths.
   - **Volume/Area Calculation:** Extracted multi-dimensional specifications (e.g., length × width × height patterns).
3. **Text Vectorization:** Utilized `TfidfVectorizer` on combined `TITLE` and `BULLET_POINTS` fields (scaling up to 6,000 max features).
4. **Modeling:** Trained an optimized **LightGBM Regressor** using a sparse matrix stack of numeric and text features, achieving a validation $R^2$ score of **~0.6410**.

## Tech Stack
- **Language:** [Python](https://www.python.org/)
- **Libraries:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Scikit-learn](https://scikit-learn.org/), [LightGBM](https://lightgbm.readthedocs.io/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
