# 🚀 AI Sales Forecasting Dashboard

An interactive, AI-powered web application built with **Streamlit** that allows users to analyze historical sales data and forecast future trends using machine learning models.

## 🌟 Features

* **Custom Data Upload:** Use the built-in default dataset (`Sample -Superstore.csv`) or upload your own CSV data.
* **Dynamic Filtering:** Filter the data in real-time by `Region` and `Category`.
* **Adjustable Granularity:** Aggregate and visualize data at `Daily`, `Weekly`, or `Monthly` levels.
* **Prophet Forecasting:** Utilizes Meta's open-source `Prophet` library to generate robust forecasts, identifying both trends and seasonality.
* **Model Comparison:** Automatically compares the accuracy of multiple algorithms (Prophet, ARIMA, and XGBoost) to find the best fit for your specific data slice.
* **KPI Metrics & Evaluation:** Displays critical metrics like Total Sales, Average Sales, Growth %, MAE, RMSE, and WMAPE-based Accuracy.
* **Export Options:** Download your forecast data directly to a CSV file for external reporting.

## 🛠️ Technology Stack

* **Frontend Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Forecasting & ML:** Prophet, Statsmodels (ARIMA), XGBoost, Scikit-Learn
* **Visualization:** Matplotlib / Streamlit native charts

## 📦 Installation & Setup

1. **Clone or download the repository** to your local machine.

2. **Navigate to the project directory** in your terminal:
   ```bash
   cd SalesForecasting-streamlit
   ```

3. **Install the required Python dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure that `xgboost` and `statsmodels` are also installed if your environment requires them for the model comparison feature).*

## 🚀 How to Run

Start the Streamlit development server by running:
```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser (usually at `http://localhost:8501`).

## 📊 Data Formatting (Custom Uploads)

If you choose to upload a custom CSV dataset, ensure your file contains at least:
* A **Date** column (e.g., `Order Date`, `Date`).
* A **Sales/Revenue** column (e.g., `Sales`, `Revenue`, `Amount`).

*Optional but recommended columns for filtering:*
* `Region`
* `Category`

The application attempts to auto-detect these columns, but keeping naming conventions standard will ensure a smooth experience.

## 💡 A Note on Forecasting Accuracy

When viewing data at highly granular levels (e.g., Daily timeframes combined with strict Region + Category filters), forecasting accuracy may naturally be lower due to the erratic and "spiky" nature of sparse sales events. For the most accurate trend forecasting, view your data at an aggregate level (e.g., Monthly granularity).
