from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load your enriched dataset once at startup
df = pd.read_csv('enriched_product_sales_2010_2025.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Unique brand and product options
brands = sorted(df['Brand'].dropna().unique())
products = sorted(df['Product_Name'].dropna().unique())

# Homepage with dropdowns
@app.route('/')
def index():
    return render_template('index.html', brands=brands, products=products)

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        brand = data.get('brand')
        product = data.get('product')
        start_date = pd.to_datetime(data.get('start_date'))
        end_date = pd.to_datetime(data.get('end_date'))

        # Filter the data for selected brand/product and date range
        filtered_df = df[
            (df['Brand'] == brand) &
            (df['Product_Name'] == product) &
            (df['timestamp'] >= start_date) &
            (df['timestamp'] <= end_date)
        ].copy()

        if filtered_df.empty:
            return jsonify({'error': 'No data found for the selected criteria.'}), 400

        # Aggregate monthly sales
        monthly_sales = filtered_df.resample('MS', on='timestamp')['Quantity_Sold'].sum().reset_index()

        # Fill missing months with 0 sales for continuity
        monthly_sales.set_index('timestamp', inplace=True)
        monthly_sales = monthly_sales.asfreq('MS', fill_value=0)
        monthly_sales.reset_index(inplace=True)

        if len(monthly_sales) < 24:
            return jsonify({'error': 'Not enough data for forecasting. Minimum 24 months required.'}), 400

        # Build and fit the Holt-Winters seasonal additive model
        model = ExponentialSmoothing(
            monthly_sales['Quantity_Sold'],
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        model_fit = model.fit()

        forecast_steps = 6
        forecast = model_fit.forecast(forecast_steps)

        # Prepare forecast dates
        last_date = monthly_sales['timestamp'].iloc[-1]
        forecast_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=forecast_steps, freq='MS')

        # Prepare response data
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
                'message': f"{product} by {brand} is projected to sell {forecast.sum():.0f} units in the next 6 months."
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
