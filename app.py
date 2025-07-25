from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import warnings
import os

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pymongo import MongoClient
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# Load environment variables from .env
load_dotenv()
MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get("DB_NAME")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def load_sales_data():
    records = list(collection.find({}))
    if not records:
        # Return empty DataFrame with expected columns if collection is empty
        return pd.DataFrame(columns=['Brand', 'Product_Name', 'timestamp', 'Quantity_Sold'])
    df = pd.DataFrame(records)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Initialize data at startup
df = load_sales_data()

# Derive min/max date if data exists
if not df.empty:
    min_dt = df['timestamp'].min().strftime('%Y-%m')
    max_dt = df['timestamp'].max().strftime('%Y-%m')
    brand_product_map = df.groupby('Brand')['Product_Name'].unique().apply(lambda x: sorted(list(x))).to_dict()
    brands = sorted(brand_product_map.keys())
else:
    min_dt = None
    max_dt = None
    brand_product_map = {}
    brands = []

@app.route('/')
def index():
    initial_brand = brands[0] if brands else None
    initial_products = brand_product_map.get(initial_brand, [])
    return render_template(
        'index.html',
        brands=brands,
        products=initial_products,
        min_date=min_dt,
        max_date=max_dt
    )

@app.route('/api/products/<brand>')
def get_products(brand):
    products = brand_product_map.get(brand, [])
    return jsonify(products)

@app.route('/api/forecast', methods=['POST'])
def forecast():
    global df  # So we can refresh data if needed

    try:
        # Force reload for fresh DB changes (remove if not needed)
        df = load_sales_data()

        data = request.get_json()
        brand = data.get('brand')
        product = data.get('product')
        start_date = pd.to_datetime(data.get('start_date'))
        end_date = pd.to_datetime(data.get('end_date'))

        filtered_df = df[
            (df['Brand'] == brand) &
            (df['Product_Name'] == product) &
            (df['timestamp'] >= start_date) &
            (df['timestamp'] <= end_date)
        ].copy()

        if filtered_df.empty:
            return jsonify({'error': 'No data found for the selected criteria. Please add data in your MongoDB cluster.'}), 400

        monthly_sales = filtered_df.resample('MS', on='timestamp')['Quantity_Sold'].sum().reset_index()
        monthly_sales.set_index('timestamp', inplace=True)
        monthly_sales = monthly_sales.asfreq('MS', fill_value=0)
        monthly_sales.reset_index(inplace=True)

        if len(monthly_sales) < 2:
            return jsonify({'error': 'Not enough data for forecasting. Minimum 2 months required.'}), 400

        if len(monthly_sales) >= 24:
            model = ExponentialSmoothing(
                monthly_sales['Quantity_Sold'],
                trend='add',
                seasonal='add',
                seasonal_periods=12
            )
            model_name = "Holt-Winters Seasonal Model"
        else:
            model = ExponentialSmoothing(
                monthly_sales['Quantity_Sold'],
                trend='add',
                seasonal=None
            )
            model_name = "Holt's Linear Trend Model"

        model_fit = model.fit()
        forecast_steps = 6
        forecast = model_fit.forecast(forecast_steps)
        forecast[forecast < 0] = 0

        last_date = monthly_sales['timestamp'].iloc[-1]
        forecast_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='MS')

        response = {
            'history': {
                'labels': monthly_sales['timestamp'].dt.strftime('%Y-%m').tolist(),
                'values': monthly_sales['Quantity_Sold'].astype(int).tolist()
            },
            'forecast': {
                'labels': forecast_dates.strftime('%Y-%m').tolist(),
                'values': forecast.round().astype(int).tolist(),
                'highs': (forecast * 1.15).round().astype(int).tolist(),
                'lows': (forecast * 0.85).round().astype(int).tolist()
            },
            'recommendation': {
                'message': f"{product} by {brand} is projected to sell {forecast.sum():.0f} units in the next 6 months.",
                'model_used': model_name
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
