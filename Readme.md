# 📚 **BuyWise - Smart Purchase Decision Platform**

## 🎯 **Overview**

**BuyWise** is an AI-powered Smart Purchase Decision Platform that helps users make informed buying decisions by analyzing product performance, sales trends, customer ratings, and market data. The platform combines advanced forecasting algorithms with machine learning to provide personalized "buy or not buy" recommendations.

### **Core Functionality**

- **AI-Powered Buy/Not Buy Recommendations** with confidence scoring
- **Sales Forecasting** using multiple ML models (Prophet, ARIMA, Holt-Winters)
- **Customer Sentiment Analysis** from ratings data
- **Personalized Recommendations** based on budget and priority preferences
- **Comprehensive Analytics Dashboard** with interactive visualizations

***

## 🏗️ **Technology Stack**

```
Frontend:    HTML5, CSS3, JavaScript, Chart.js
Backend:     Python Flask
Database:    MongoDB
AI/ML:       scikit-learn, Prophet, ARIMA, Holt-Winters
Processing:  pandas, numpy
Visualization: Chart.js with zoom/pan plugins
```


***

## 🚀 **Key Features**

### **1. AI-Powered Purchase Recommendations**

- **Machine Learning Classifier**: Random Forest trained on ratings and sales data
- **Confidence Scoring**: Provides percentage confidence in recommendations
- **Personalization Engine**: Adjusts recommendations based on user preferences
- **Budget Integration**: Considers user budget constraints
- **Priority Factors**: Rating-focused, Value-focused, Popularity-focused, or Balanced


### **2. Advanced Sales Forecasting**

- **Multiple AI Models**: Prophet, ARIMA, Holt-Winters, Auto-ARIMA
- **Automatic Model Selection**: Chooses best performing model via backtesting
- **6-Month Predictions**: Future sales forecasts with confidence intervals
- **Seasonal Pattern Analysis**: Identifies cyclical trends and patterns


### **3. Comprehensive Analytics Dashboard**

- **Sales Forecasting**: Historical analysis, future predictions, seasonal patterns
- **Revenue Analytics**: Trends, growth analysis, quantity correlations
- **Performance Analysis**: Year-over-year growth, price-sales correlations
- **Customer Ratings \& AI**: Rating trends, sentiment analysis, AI recommendations


### **4. User Personalization System**

```
🎯 Personalization Options:
- Budget: Optional budget constraint for recommendations
- Priority: Balanced | Rating-focused | Value-focused | Popularity-focused
- Dynamic ML Adjustments: Confidence scores adapt to user preferences
```


***

## 📋 **Installation \& Setup**

### **Prerequisites**

```bash
Python 3.8+
MongoDB 4.0+
```


### **Installation Steps**

1. **Clone the repository**
```bash
git clone https://github.com/summui/project.git
cd forecastfuturebuy
```

2. **Install Python dependencies**
```bash
pip install flask pandas numpy scikit-learn prophet statsmodels pymongo python-dotenv joblib
```

3. **Optional: Install pmdarima for Auto-ARIMA**
```bash
pip install pmdarima
```

4. **Create environment file**
```bash
# Create .env file with your MongoDB credentials
MONGODB_URI=your_mongodb_connection_string
DB_NAME=your_database_name
COLLECTION_NAME=your_collection_name
```

5. **Run the application**
```bash
python app.py
```

6. **Access the application**
```
Open browser: http://localhost:5000
Health check: http://localhost:5000/api/health
```


***

## 📊 **Required Data Schema**

Your MongoDB collection should contain documents with this structure:

```json
{
  "Brand": "JBL",
  "Product_Name": "Headphones", 
  "Model": "Tune 510BT",
  "timestamp": "2024-01-01T00:00:00Z",
  "Quantity_Sold": 150,
  "Unit_Price": 49.99,
  "Revenue": 7498.50,
  "Year": 2024,
  "Month": "January",
  "Product_ID": "JBL_HEAD_001",
  "Rating": 4.2
}
```


***

## 👥 **User Guide**

### **Step 1: Basic Product Selection**

1. **Select Brand**: Choose from available brands in dropdown
2. **Select Product**: Product list updates based on brand selection
3. **Select Model**: Model options appear after product selection
4. **Set Date Range**: Choose analysis period (start and end dates)

### **Step 2: Personalization Settings**

#### **Budget (Optional)**

```
Enter your budget limit (e.g., $500)
- System warns if product may exceed budget
- Enhances "Best Value" priority recommendations
- Considered in AI recommendation summary
```


#### **Priority Factor**

```
Balanced: Equal weight to ratings and sales performance
High Rating Priority: +10% boost for products with 4+ star ratings
Best Value Priority: +15% boost for products under your budget
Popularity Priority: +10% boost for top-selling products
```


### **Step 3: Generate Analysis**

Click **"Generate Smart Analysis"** to trigger:

- Multi-model sales forecasting
- Rating and sentiment analysis
- AI-powered buy recommendation with confidence score
- Personalized insights based on your preferences


### **Step 4: Interpreting Results**

#### **Smart Purchase Analysis**

```
✅ Recommended Purchase (85.3% confidence)
❌ Not Recommended (32.1% confidence)
⚠️ Insufficient data for prediction
```


#### **AI Recommendation Summary**

Contextual explanation with reasoning:

- Confidence score breakdown
- Sentiment analysis insights
- Budget consideration notes
- Priority factor effects

***

## 🔧 **API Documentation**

### **Core Endpoints**

#### **GET /**

Main dashboard interface

#### **GET /api/products/{brand}**

Returns products for specified brand

```json
["Headphones", "Speakers", "Earbuds"]
```


#### **GET /api/models/{brand}/{product}**

Returns models for specified brand and product

```json
["Tune 510BT", "Live 460NC", "Clip 3"]
```


#### **POST /api/forecast**

Generates sales forecast and analysis

```json
{
  "brand": "JBL",
  "product": "Headphones",
  "model": "Tune 510BT",
  "start_date": "2023-01-01",
  "end_date": "2025-08-01",
  "budget": 500.0,        // Optional
  "priority": "rating"    // Optional: balanced|rating|value|popularity
}
```


#### **POST /api/ratings**

Enhanced ratings analysis with buy predictions

```json
{
  "brand": "JBL",
  "product": "Headphones", 
  "model": "Tune 510BT",
  "start_date": "2023-01-01",
  "end_date": "2025-08-01",
  "budget": 500.0,        // Optional
  "priority": "rating"    // Optional
}
```

**Response includes:**

- Rating trends and distribution
- Sentiment breakdown
- Buy probability and recommendation
- Ratings vs sales correlation


#### **POST /api/recommendation**

Combined forecast and buy recommendation

```json
{
  "forecast": {
    "labels": ["2025-09", "2025-10", ...],
    "values": [120, 135, 150, ...]
  },
  "buy_recommendation": "Buy",
  "buy_probability": 0.85,
  "forecast_model_used": "Prophet"
}
```


#### **GET /api/health**

System health check

```json
{
  "status": "healthy",
  "ml_model_loaded": true,
  "data_points": 5000,
  "brands_available": 15
}
```


***

## 🧠 **Machine Learning Implementation**

### **Buy/Not Buy Classification**

#### **Training Logic**

```python
# Feature Engineering
Features: ['Rating', 'Quantity_Sold']
Target: Buy = 1 if Rating >= 4, else 0

# Model Training
RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
```


#### **Personalization Adjustments**

```python
base_prediction = model.predict_proba(features)[0][1]

# Priority-based adjustments
if priority == 'rating' and avg_rating >= 4.0:
    confidence += 0.1  # 10% boost
elif priority == 'value' and price < budget:
    confidence += 0.15  # 15% boost  
elif priority == 'popularity' and sales > 75th_percentile:
    confidence += 0.1  # 10% boost

final_recommendation = 'Buy' if confidence > 0.5 else 'Do not buy'
```


### **Forecasting Models**

#### **Model Selection Process**

1. **Available Models**: Prophet, ARIMA, Holt-Winters, Auto-ARIMA
2. **Backtesting**: Each model tested on historical data
3. **RMSE Calculation**: Performance measured via Root Mean Square Error
4. **Auto Selection**: Best performing model chosen automatically

#### **Forecast Output**

- **Point Predictions**: Most likely future values
- **Confidence Intervals**: Upper and lower bounds (±15%)
- **Model Attribution**: Shows which model was selected
- **Performance Metrics**: RMSE scores for model comparison

***

## 📊 **Dashboard Tabs**

### **Sales Forecasting Tab**

- **Historical Sales Analysis**: Past performance with moving averages
- **Future Sales Forecast**: 6-month predictions with confidence bands
- **Seasonal Patterns**: Interactive heatmap showing monthly trends
- **Revenue Overview**: Combined historical and forecasted revenue


### **Revenue Analytics Tab**

- **Revenue Trend Analysis**: Revenue over time with trend lines
- **Monthly Revenue Growth**: Month-over-month percentage changes
- **Revenue vs. Quantity**: Scatter plot correlation analysis
- **Cumulative Revenue**: Running total revenue progression
- **Average Revenue per Unit**: Price trend analysis


### **Performance Analysis Tab**

- **Year-over-Year Growth**: Annual growth rate comparison
- **Price vs. Sales Correlation**: Price elasticity visualization


### **Customer Ratings \& Buy AI Tab**

- **Rating Trends Over Time**: Monthly rating progression
- **Rating Distribution**: Histogram of 1-5 star ratings
- **Ratings vs. Sales Performance**: Rating-sales correlation scatter plot
- **Sentiment Analysis**: Positive/Neutral/Negative breakdown pie chart
- **AI Purchase Recommendation**: Contextual AI explanation and reasoning

***

## 🎯 **Use Cases**

### **For Consumers**

- **Product Research**: Get AI insights before purchasing decisions
- **Budget Planning**: Set spending constraints for personalized recommendations
- **Preference Matching**: Prioritize rating quality, value, or popularity
- **Market Timing**: Understand optimal purchase timing


### **For Businesses**

- **Inventory Planning**: 6-month sales forecasts for stock management
- **Market Analysis**: Customer sentiment and rating trend insights
- **Performance Monitoring**: Track YoY growth and revenue patterns
- **Competitive Intelligence**: Analyze product performance across categories

***

## 🔍 **How Personalization Works**

### **Budget Feature**

1. **User Input**: Optional budget field (e.g., \$500)
2. **Price Calculation**: Average product price from historical data
3. **Budget Validation**: Comparison of price vs budget
4. **Recommendation Impact**:
    - Under budget + "Value" priority = +15% confidence boost
    - Over budget = Warning message in recommendation
    - Budget consideration included in AI summary

### **Priority Factors**

#### **High Rating Priority**

- **Logic**: Boost products with average rating ≥ 4.0 by 10%
- **Use Case**: Quality-focused customers
- **AI Explanation**: "Emphasizes customer satisfaction scores"


#### **Best Value Priority**

- **Logic**: Boost products under user budget by 15%
- **Use Case**: Budget-conscious customers
- **AI Explanation**: "Focuses on price-to-value optimization"


#### **Popularity Priority**

- **Logic**: Boost products in top 25% sales volume by 10%
- **Use Case**: Trend-following customers
- **AI Explanation**: "Considers market demand patterns"


#### **Balanced Priority**

- **Logic**: No adjustments, pure ML prediction
- **Use Case**: Algorithm-trusting customers
- **AI Explanation**: "Equal weight to all performance factors"

***

## 🔧 **Troubleshooting**

### **Common Issues \& Solutions**

#### **JavaScript Errors**

```
Error: "renderHistorical not defined"
Solution: Ensure complete HTML template is used with all chart functions

Error: "sentimentData.values.toFixed is not a function" 
Solution: Updated with null safety checks (already fixed)
```


#### **ML Model Issues**

```
Error: "Feature names should match those that were passed during fit"
Solution: Delete buy_classifier.joblib file to force model retraining

Warning: "Not enough data to train ML model"
Solution: Ensure at least 10 records with Rating and Quantity_Sold data
```


#### **Database Connection**

```
Error: "Missing MongoDB configuration"
Solution: Verify .env file contains MONGODB_URI, DB_NAME, COLLECTION_NAME

Error: "No data found for selected criteria"
Solution: Check data exists for selected brand/product/model combination
```


### **Performance Optimization**

- **Data Loading**: Only reloads when version changes
- **Model Training**: Only retrains when new data detected
- **Chart Rendering**: Optimized for large datasets with downsampling
- **Memory Management**: Efficient pandas operations with garbage collection


### **Health Monitoring**

```bash
# Check system status
curl http://localhost:5000/api/health

# Expected response
{
  "status": "healthy",
  "ml_model_loaded": true,
  "data_points": 5000,
  "brands_available": 15
}
```


***

## 📈 **Future Enhancement Opportunities**

### **Advanced AI Features**

- **NLP Integration**: Process customer review text for deeper sentiment analysis
- **Image Analysis**: Analyze product photos using computer vision
- **Price Prediction**: Forecast future pricing trends
- **Competitor Analysis**: Multi-brand comparison capabilities


### **Technical Improvements**

- **Caching Layer**: Redis for improved response times
- **API Authentication**: User authentication and rate limiting
- **Real-time Updates**: WebSocket integration for live data
- **Mobile App**: React Native or Flutter implementation


### **Business Features**

- **User Accounts**: Save preferences and purchase history
- **Watchlists**: Track multiple products over time
- **Price Alerts**: Notifications for optimal purchase timing
- **Export Capabilities**: PDF reports and CSV data exports

***

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

***

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

***

## 📞 **Support**

For support and questions:

- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the API documentation for integration help

***

**Built with ❤️ using Python, Flask, and AI/ML technologies**

