# E-Commerce Sales Analysis and Demand Prediction

## Project
This project analyzes transaction-level e-commerce data and adds a machine-learning component for monthly revenue prediction.

## Dataset
The code is designed for the commonly used Online Retail transaction dataset. Place the dataset beside the Python script as either:
- `Online Retail.xlsx`
- `Online Retail.csv`
- `Online_Retail.csv`

The dataset should contain fields equivalent to InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID and Country.

## Run
```bash
pip install -r requirements.txt
python ecommerce_analysis.py
```

## Required visualizations
1. Monthly revenue trend
2. Top 10 products by revenue
3. Top 10 countries by revenue
4. Order-value distribution
5. Revenue by day of week

An additional actual-vs-predicted revenue chart is produced when enough monthly observations exist.

## Machine learning
A Random Forest Regressor predicts monthly revenue using calendar/time features. Evaluation uses MAE, RMSE and R² on a chronological holdout.

## Important
Run the code on the actual dataset before final submission if numeric results are required. The report does not invent accuracy, revenue or model metrics.
