"""
E-Commerce Sales Analysis and Demand Prediction
Internship Project

Dataset expected:
    Online Retail.xlsx  OR  Online Retail.csv

The commonly used Online Retail dataset contains transaction-level e-commerce
records. The script accepts either Excel or CSV input.

Run:
    pip install -r requirements.txt
    python ecommerce_analysis.py

Outputs are written to outputs/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

def load_data():
    excel = BASE / "Online Retail.xlsx"
    csv = BASE / "Online Retail.csv"
    csv2 = BASE / "Online_Retail.csv"

    if excel.exists():
        return pd.read_excel(excel)
    if csv.exists():
        return pd.read_csv(csv, encoding="latin1")
    if csv2.exists():
        return pd.read_csv(csv2, encoding="latin1")
    raise FileNotFoundError(
        "Place the Online Retail dataset as 'Online Retail.xlsx' or "
        "'Online Retail.csv' beside this script."
    )

def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()

def main():
    df = load_data().copy()

    # Standardize column names used by the common Online Retail dataset.
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "").replace("_", "")
        if key == "invoiceno": rename[c] = "InvoiceNo"
        elif key == "stockcode": rename[c] = "StockCode"
        elif key == "description": rename[c] = "Description"
        elif key == "quantity": rename[c] = "Quantity"
        elif key == "invoicedate": rename[c] = "InvoiceDate"
        elif key == "unitprice": rename[c] = "UnitPrice"
        elif key == "customerid": rename[c] = "CustomerID"
        elif key == "country": rename[c] = "Country"
    df = df.rename(columns=rename)

    required = ["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","Country"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate","Quantity","UnitPrice","InvoiceNo"])
    df = df.drop_duplicates()

    # Exclude cancellations and invalid negative/zero sales for sales analysis.
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    sales = df[~df["InvoiceNo"].str.upper().str.startswith("C")].copy()
    sales = sales[(sales["Quantity"] > 0) & (sales["UnitPrice"] > 0)]
    sales["Revenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["YearMonth"] = sales["InvoiceDate"].dt.to_period("M").astype(str)
    sales["Month"] = sales["InvoiceDate"].dt.month
    sales["DayOfWeek"] = sales["InvoiceDate"].dt.day_name()
    sales["Hour"] = sales["InvoiceDate"].dt.hour

    # Summary
    summary = pd.DataFrame({
        "Metric": [
            "Rows after cleaning", "Unique invoices", "Unique products",
            "Unique countries", "Unique customers", "Total revenue",
            "Average order value"
        ],
        "Value": [
            len(sales), sales["InvoiceNo"].nunique(),
            sales["StockCode"].nunique(), sales["Country"].nunique(),
            sales["CustomerID"].nunique() if "CustomerID" in sales else np.nan,
            sales["Revenue"].sum(),
            sales.groupby("InvoiceNo")["Revenue"].sum().mean()
        ]
    })
    summary.to_csv(OUT / "dataset_summary.csv", index=False)

    # 1. Monthly revenue trend
    monthly = sales.groupby("YearMonth")["Revenue"].sum()
    monthly.plot(kind="line", marker="o", title="Monthly E-Commerce Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    savefig("01_monthly_revenue_trend.png")
    monthly.rename("Revenue").to_csv(OUT / "monthly_revenue.csv")

    # 2. Top product descriptions
    top_products = sales.groupby("Description")["Revenue"].sum().nlargest(10).sort_values()
    top_products.plot(kind="barh", title="Top 10 Products by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Product")
    savefig("02_top_products_by_revenue.png")
    top_products.rename("Revenue").to_csv(OUT / "top_products.csv")

    # 3. Revenue by country
    country = sales.groupby("Country")["Revenue"].sum().nlargest(10).sort_values()
    country.plot(kind="barh", title="Top 10 Countries by Revenue")
    plt.xlabel("Revenue")
    plt.ylabel("Country")
    savefig("03_top_countries_by_revenue.png")
    country.rename("Revenue").to_csv(OUT / "country_revenue.csv")

    # 4. Order value distribution
    order_value = sales.groupby("InvoiceNo")["Revenue"].sum()
    order_value.plot(kind="hist", bins=40, title="Distribution of Order Values")
    plt.xlabel("Order Value")
    plt.ylabel("Number of Orders")
    savefig("04_order_value_distribution.png")
    order_value.describe().to_csv(OUT / "order_value_summary.csv")

    # 5. Revenue by day of week
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = sales.groupby("DayOfWeek")["Revenue"].sum().reindex(day_order).dropna()
    dow.plot(kind="bar", title="Revenue by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Revenue")
    plt.xticks(rotation=30)
    savefig("05_revenue_by_day_of_week.png")
    dow.rename("Revenue").to_csv(OUT / "day_of_week_revenue.csv")

    # Observations/insights data tables
    observations = pd.DataFrame({
        "Observation": [
            "Monthly revenue varies over time and can reveal seasonal or promotional patterns.",
            "A small group of products may contribute a large share of total revenue.",
            "Revenue is concentrated across a limited number of countries.",
            "Order values are not evenly distributed and may contain a long tail.",
            "Revenue differs by day of week, indicating possible timing effects."
        ]
    })
    observations.to_csv(OUT / "five_observations.csv", index=False)

    insights = pd.DataFrame({
        "Insight": [
            "Revenue trends can support inventory and campaign planning.",
            "High-revenue products can receive closer availability and merchandising attention.",
            "Country-level concentration can inform geographic marketing priorities.",
            "Order-value distribution can support customer segmentation and promotion design.",
            "Day-of-week patterns can help schedule campaigns and operational capacity."
        ]
    })
    insights.to_csv(OUT / "five_insights.csv", index=False)

    hypotheses = pd.DataFrame({
        "Hypothesis": [
            "H1: Monthly revenue changes significantly across time periods.",
            "H2: A small number of products account for a disproportionate share of revenue.",
            "H3: Average order value differs across countries."
        ],
        "Suggested_Test": [
            "Time-series comparison and, where appropriate, statistical comparison of periods.",
            "Concentration analysis such as cumulative revenue share/Pareto analysis.",
            "Compare order-value distributions by selected countries using suitable statistical tests."
        ]
    })
    hypotheses.to_csv(OUT / "three_hypotheses.csv", index=False)

    # -----------------------------
    # Demand/revenue prediction
    # Forecast monthly revenue using calendar features.
    # -----------------------------
    forecast = monthly.reset_index()
    forecast.columns = ["YearMonth", "Revenue"]
    dates = pd.to_datetime(forecast["YearMonth"] + "-01")
    forecast["Year"] = dates.dt.year
    forecast["MonthNum"] = dates.dt.month
    forecast["TimeIndex"] = np.arange(len(forecast))

    X = forecast[["Year", "MonthNum", "TimeIndex"]]
    y = forecast["Revenue"]

    if len(forecast) >= 12:
        split = max(1, int(len(forecast) * 0.8))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=2
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = pd.DataFrame([{
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred)
        }])
        metrics.to_csv(OUT / "revenue_prediction_metrics.csv", index=False)

        prediction_table = pd.DataFrame({
            "YearMonth": forecast["YearMonth"].iloc[split:].values,
            "ActualRevenue": y_test.values,
            "PredictedRevenue": pred
        })
        prediction_table.to_csv(OUT / "revenue_predictions.csv", index=False)

        plt.plot(prediction_table["YearMonth"], prediction_table["ActualRevenue"], label="Actual")
        plt.plot(prediction_table["YearMonth"], prediction_table["PredictedRevenue"], label="Predicted")
        plt.title("Actual vs Predicted Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.legend()
        savefig("06_actual_vs_predicted_revenue.png")
    else:
        (OUT / "revenue_prediction_note.txt").write_text(
            "Fewer than 12 monthly observations were available; prediction evaluation was skipped.",
            encoding="utf-8"
        )

    print("E-COMMERCE PROJECT COMPLETE")
    print(f"Clean sales rows: {len(sales):,}")
    print(f"Total revenue: {sales['Revenue'].sum():,.2f}")
    print(f"Outputs saved in: {OUT}")

if __name__ == "__main__":
    main()
