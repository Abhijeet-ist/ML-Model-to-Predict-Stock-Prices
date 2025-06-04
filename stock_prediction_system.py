import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, csv_file_paths):
        """
        Initialize the Stock Prediction System
        
        Args:
            csv_file_paths (str or list): Path(s) to the CSV file(s) containing stock data
                                         Can be a single file path or a list of file paths
        """
        # Handle both single path and list of paths
        if isinstance(csv_file_paths, str):
            self.csv_file_paths = [csv_file_paths]
        else:
            self.csv_file_paths = csv_file_paths
            
        self.df = None
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'Close'
        
    def load_and_preprocess_data(self):
        """Load and preprocess the stock data from multiple CSV files"""
        try:
            all_dataframes = []
            
            for csv_file_path in self.csv_file_paths:
                print(f"Loading data from: {csv_file_path}")
                # Load the data
                df = pd.read_csv(csv_file_path)
                
                # Clean column names (remove extra spaces)
                df.columns = df.columns.str.strip()
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
                
                # Remove commas and convert numeric columns
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (₹ Cr)']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                
                # Handle missing values
                df = df.dropna()
                
                # Store processed dataframe
                all_dataframes.append(df)
                print(f"  - Loaded {df.shape[0]} rows from {csv_file_path}")
            
            # Combine all dataframes
            if len(all_dataframes) > 0:
                self.df = pd.concat(all_dataframes)
                # Sort by date
                self.df = self.df.sort_values('Date').reset_index(drop=True)
                # Remove duplicate dates if any
                self.df = self.df.drop_duplicates(subset=['Date']).reset_index(drop=True)
                
                print(f"\nCombined data shape: {self.df.shape}")
                print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
                return True
            else:
                print("No valid data found in any of the CSV files.")
                return False
                
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_technical_indicators(self):
        """Create technical indicators for better prediction"""
        
        # Simple Moving Averages
        self.df['SMA_5'] = self.df['Close'].rolling(window=5).mean()
        self.df['SMA_10'] = self.df['Close'].rolling(window=10).mean()
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        
        # Exponential Moving Averages
        self.df['EMA_12'] = self.df['Close'].ewm(span=12).mean()
        self.df['EMA_26'] = self.df['Close'].ewm(span=26).mean()
        
        # MACD
        self.df['MACD'] = self.df['EMA_12'] - self.df['EMA_26']
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9).mean()
        
        # RSI (Relative Strength Index)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.df['BB_middle'] = self.df['Close'].rolling(window=20).mean()
        bb_std = self.df['Close'].rolling(window=20).std()
        self.df['BB_upper'] = self.df['BB_middle'] + (bb_std * 2)
        self.df['BB_lower'] = self.df['BB_middle'] - (bb_std * 2)
        self.df['BB_width'] = self.df['BB_upper'] - self.df['BB_lower']
        
        # Price-based features
        self.df['Price_Range'] = self.df['High'] - self.df['Low']
        self.df['Price_Change'] = self.df['Close'] - self.df['Open']
        self.df['Price_Change_Pct'] = (self.df['Close'] - self.df['Open']) / self.df['Open'] * 100
        
        # Volume-based features
        self.df['Volume_SMA'] = self.df['Shares Traded'].rolling(window=10).mean()
        self.df['Volume_Ratio'] = self.df['Shares Traded'] / self.df['Volume_SMA']
        
        # Volatility
        self.df['Volatility'] = self.df['Close'].rolling(window=10).std()
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            self.df[f'Close_lag_{lag}'] = self.df['Close'].shift(lag)
            self.df[f'Volume_lag_{lag}'] = self.df['Shares Traded'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            self.df[f'Close_rolling_mean_{window}'] = self.df['Close'].rolling(window=window).mean()
            self.df[f'Close_rolling_std_{window}'] = self.df['Close'].rolling(window=window).std()
            self.df[f'Close_rolling_max_{window}'] = self.df['Close'].rolling(window=window).max()
            self.df[f'Close_rolling_min_{window}'] = self.df['Close'].rolling(window=window).min()
        
        # Time-based features
        self.df['Day_of_Week'] = self.df['Date'].dt.dayofweek
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Drop rows with NaN values created by rolling calculations
        self.df = self.df.dropna()
        
        print(f"Technical indicators created. New shape: {self.df.shape}")
    
    def prepare_features(self):
        """Prepare feature matrix for machine learning"""
        
        # Define feature columns (exclude Date and target variable)
        exclude_cols = ['Date', 'Close']
        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        
        print(f"Total features: {len(self.feature_columns)}")
        print("Feature columns:", self.feature_columns[:10], "...")
        
        return self.df[self.feature_columns], self.df[self.target_column]
    
    def train_models(self, test_size=0.2, random_state=42):
        """Train multiple machine learning models"""
        
        X, y = self.prepare_features()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale the features
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        X_train_minmax = self.scalers['minmax'].fit_transform(X_train)
        X_test_minmax = self.scalers['minmax'].transform(X_test)
        
        # Define models
        model_configs = {
            'Linear Regression': {
                'model': LinearRegression(),
                'data': (X_train_scaled, X_test_scaled, y_train, y_test)
            },
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
                'data': (X_train, X_test, y_train, y_test)
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
                'data': (X_train, X_test, y_train, y_test)
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
                'data': (X_train, X_test, y_train, y_test)
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=100, gamma=0.1),
                'data': (X_train_scaled, X_test_scaled, y_train, y_test)
            },
            'Neural Network': {
                'model': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state),
                'data': (X_train_minmax, X_test_minmax, y_train, y_test)
            }
        }
        
        # Train and evaluate models
        results = {}
        
        print("Training models...")
        for name, config in model_configs.items():
            print(f"\nTraining {name}...")
            
            model = config['model']
            X_tr, X_te, y_tr, y_te = config['data']
            
            # Train the model
            model.fit(X_tr, y_tr)
            
            # Make predictions
            y_pred = model.predict(X_te)
            
            # Calculate metrics
            mse = mean_squared_error(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)
            rmse = np.sqrt(mse)
            
            # Store results
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'actual': y_te
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        self.models = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def evaluate_models(self):
        """Evaluate and compare all models with more sophisticated model selection"""
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.models.items():
            comparison_data.append({
                'Model': name,
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'R²': result['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add a composite score for better model selection
        # This takes into account both accuracy (R²) and error metrics (RMSE, MAE)
        # Normalize metrics first
        max_rmse = comparison_df['RMSE'].max()
        max_mae = comparison_df['MAE'].max()
        
        comparison_df['RMSE_norm'] = 1 - (comparison_df['RMSE'] / max_rmse)
        comparison_df['MAE_norm'] = 1 - (comparison_df['MAE'] / max_mae)
        
        # Composite score (weighted average of R², normalized RMSE, and normalized MAE)
        comparison_df['Composite_Score'] = (
            0.5 * comparison_df['R²'] + 
            0.3 * comparison_df['RMSE_norm'] + 
            0.2 * comparison_df['MAE_norm']
        )
        
        # Sort by composite score
        comparison_df = comparison_df.sort_values('Composite_Score', ascending=False)
        
        # Add a reliability flag for suspiciously perfect models
        comparison_df['Reliable'] = 'Yes'
        comparison_df.loc[comparison_df['R²'] > 0.99, 'Reliable'] = 'Check for overfitting'
        
        # Best model (considering reliability)
        best_models = comparison_df[comparison_df['Reliable'] == 'Yes']
        if len(best_models) > 0:
            best_model_name = best_models.iloc[0]['Model']
        else:
            # If all models are flagged, still pick the best one but with a warning
            best_model_name = comparison_df.iloc[0]['Model']
            print("Warning: All top models show signs of potential overfitting")
        
        # Print comparison table
        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        display_cols = ['Model', 'RMSE', 'MAE', 'R²', 'Composite_Score', 'Reliable']
        print(comparison_df[display_cols].to_string(index=False, float_format='%.4f'))
        
        print(f"\nBest performing model: {best_model_name}")
        
        # Print model recommendations
        print("\nMODEL RECOMMENDATIONS:")
        best_score = comparison_df.iloc[0]['Composite_Score']
        second_score = comparison_df.iloc[1]['Composite_Score'] if len(comparison_df) > 1 else 0
        
        if best_score > 0.8:
            print("✅ Model performance is good for stock prediction")
        elif best_score > 0.6:
            print("⚠️ Model performance is acceptable but could be improved")
        else:
            print("❌ Model performance is poor - predictions may be unreliable")
            
        if best_score - second_score < 0.05 and len(comparison_df) > 1:
            print(f"📊 Consider ensemble methods: {comparison_df.iloc[0]['Model']} and {comparison_df.iloc[1]['Model']} have similar performance")
        
        return comparison_df, best_model_name
    
    def plot_predictions(self, model_name=None, n_points=100):
        """Plot actual vs predicted values with detailed explanations"""
        
        if model_name is None:
            # Use the best model
            comparison_df, model_name = self.evaluate_models()
            model_name = comparison_df.iloc[0]['Model']
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        result = self.models[model_name]
        y_actual = result['actual']
        y_pred = result['predictions']
        
        # Limit points for cleaner visualization
        if len(y_actual) > n_points:
            indices = np.linspace(0, len(y_actual)-1, n_points, dtype=int)
            y_actual_plot = y_actual.iloc[indices]
            y_pred_plot = y_pred[indices]
        else:
            y_actual_plot = y_actual
            y_pred_plot = y_pred
        
        # Calculate key metrics for analysis
        mse = result['mse']
        mae = result['mae']
        r2 = result['r2']
        rmse = result['rmse']
        
        # Create the main figure
        plt.figure(figsize=(16, 12))
        
        # Plot 1: Actual vs Predicted Time Series
        plt.subplot(2, 2, 1)
        plt.plot(y_actual_plot.values, label='Actual', alpha=0.7, linewidth=2)
        plt.plot(y_pred_plot, label='Predicted', alpha=0.7, linewidth=2)
        plt.title(f'{model_name} - Actual vs Predicted Stock Prices', fontsize=12, fontweight='bold')
        plt.xlabel('Time Points')
        plt.ylabel('Stock Price (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Scatter Plot
        plt.subplot(2, 2, 2)
        plt.scatter(y_actual, y_pred, alpha=0.6)
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        plt.xlabel('Actual Stock Price (₹)')
        plt.ylabel('Predicted Stock Price (₹)')
        plt.title(f'{model_name} - Prediction Accuracy', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        plt.subplot(2, 2, 3)
        residuals = y_actual.values - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Stock Price (₹)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title(f'{model_name} - Residual Analysis', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error (₹)')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Error Distribution', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add overall analysis text at the bottom
        plt.figtext(0.5, 0.01, f"""
        📊 MODEL PERFORMANCE ANALYSIS
        
        Model: {model_name}
        R² Score: {r2:.4f} (higher is better, 1.0 is perfect prediction)
        RMSE: ₹{rmse:.2f} (lower is better, represents average prediction error in ₹)
        MAE: ₹{mae:.2f} (lower is better, represents average absolute prediction error in ₹)
        
        GRAPH EXPLANATIONS:
        
        1. Actual vs Predicted: Shows how well the model's predictions (orange) follow the actual prices (blue) over time.
           - Tight tracking indicates good prediction accuracy
           - Gaps indicate areas where the model struggles to predict accurately
        
        2. Prediction Accuracy: Each point represents an actual price vs its prediction.
           - Points closer to the red line indicate more accurate predictions
           - Clustering of points shows where the model is most accurate
        
        3. Residual Analysis: Shows prediction errors against predicted values.
           - Points scattered randomly around zero line indicate a good model
           - Patterns indicate systematic errors in the model
        
        4. Error Distribution: Shows the distribution of prediction errors.
           - Ideally centered at zero with narrow spread
           - Skewness indicates bias toward over or under-prediction
        
        INSIGHT: This model {'has good predictive power' if r2 > 0.7 else 'has limited predictive power'} with an average error of ₹{rmse:.2f}.
        {'Consider trading decisions with confidence.' if r2 > 0.8 else 'Use caution when making trading decisions based on these predictions.'}
        """, ha='center', va='bottom', bbox=dict(boxstyle="round,pad=1", facecolor='lightblue', alpha=0.1), fontsize=11)
        
        plt.subplots_adjust(bottom=0.25)  # Make space for the text
        plt.show()
    
    def get_feature_importance(self, model_name='Random Forest'):
        """Get and plot feature importance"""
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top 20 Feature Importances')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance_df.head(10).to_string(index=False))
            
            return feature_importance_df
        else:
            print(f"Model {model_name} doesn't support feature importance!")
    
    def predict_future(self, model_name=None, days=5):
        """Predict future stock prices for the next 'days' trading days"""
        
        if model_name is None:
            comparison_df, model_name = self.evaluate_models()
            # If Linear Regression shows perfect R^2, it might be overfitting
            # Choose the next best model
            if model_name == 'Linear Regression' and comparison_df.iloc[0]['R²'] > 0.99:
                model_name = comparison_df.iloc[1]['Model']
                print(f"Notice: Switching to {model_name} to avoid potential overfitting")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        model = self.models[model_name]['model']
        current_price = self.df['Close'].iloc[-1]
        
        print(f"\n{model_name} Predictions for Next {days} Trading Days:")
        print(f"Current Price: ₹{current_price:.2f}")
        
        # Store predictions for plotting
        future_dates = pd.date_range(start=self.df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days, freq='B')
        future_prices = []
        
        # Create a copy of the last row for iterative prediction
        last_data = self.df.iloc[-1:].copy()
        
        for i in range(days):
            # Extract features from the last available data
            if i == 0:
                # For first prediction, use actual data
                features = self.df[self.feature_columns].iloc[-1:].values
            else:
                # For subsequent predictions, use previously predicted data
                # This is a simplified approach - in a real scenario, you'd need to update all features
                features = last_data[self.feature_columns].values
            
            # Apply appropriate scaling
            if model_name in ['Linear Regression', 'SVR']:
                features_scaled = self.scalers['standard'].transform(features)
                prediction = model.predict(features_scaled)[0]
            elif model_name == 'Neural Network':
                features_scaled = self.scalers['minmax'].transform(features)
                prediction = model.predict(features_scaled)[0]
            else:
                prediction = model.predict(features)[0]
            
            # Store prediction
            future_prices.append(prediction)
            
            # Update last_data for next iteration (simplified)
            # In a real scenario, you'd need to update all features properly
            last_data['Close'] = prediction
            # Update some basic features that depend directly on Close
            if i > 0:  # Skip first iteration as we already have real data
                # Update price-based features
                last_data['SMA_5'] = prediction  # Simplified
                last_data['EMA_12'] = prediction  # Simplified
                last_data['EMA_26'] = prediction  # Simplified
            
            # Calculate change
            change = prediction - current_price
            change_pct = (change / current_price) * 100
            
            print(f"Day {i+1}: ₹{prediction:.2f} | Change: ₹{change:.2f} ({change_pct:.2f}%)")
        
        # Plot future predictions
        self.plot_future_predictions(future_dates, future_prices, model_name)
        
        return future_prices
    
    def plot_future_predictions(self, future_dates, future_prices, model_name):
        """Plot future price predictions with actionable insights"""
        current_price = self.df['Close'].iloc[-1]
        
        # Get historical dates and prices for context
        historical_dates = self.df['Date'].iloc[-30:].tolist()
        historical_prices = self.df['Close'].iloc[-30:].tolist()
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Plot main prediction chart
        plt.subplot(2, 1, 1)
        plt.plot(historical_dates, historical_prices, 'b-', label='Historical Prices', linewidth=2)
        plt.plot(future_dates, future_prices, 'r--', marker='o', label='Predicted Prices', linewidth=2)
        
        # Add a marker for the current price
        plt.axhline(y=current_price, color='green', linestyle='--', alpha=0.7)
        plt.text(historical_dates[-1], current_price, f' Current: ₹{current_price:.2f}', 
                 verticalalignment='bottom', fontsize=10)
        
        # Format the plot
        plt.title(f'Stock Price Prediction - Next {len(future_prices)} Trading Days', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (₹)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Add annotations for predicted values
        for i, (date, price) in enumerate(zip(future_dates, future_prices)):
            change = price - current_price
            change_pct = (change / current_price) * 100
            color = 'green' if price > current_price else 'red'
            plt.annotate(f'₹{price:.2f} ({change_pct:.1f}%)', 
                         (date, price), 
                         textcoords="offset points",
                         xytext=(0,10), 
                         ha='center',
                         color=color,
                         fontweight='bold')
        
        # Determine trend and recommendation
        last_prediction = future_prices[-1]
        overall_change = last_prediction - current_price
        overall_change_pct = (overall_change / current_price) * 100
        
        # Calculate trend strength
        trend_strength = abs(overall_change_pct)
        if trend_strength < 1:
            strength_desc = "weak"
        elif trend_strength < 3:
            strength_desc = "moderate"
        else:
            strength_desc = "strong"
            
        # Determine if trend is consistent
        price_changes = [future_prices[i] - future_prices[i-1] for i in range(1, len(future_prices))]
        consistent = all(change > 0 for change in price_changes) or all(change < 0 for change in price_changes)
        consistency_desc = "consistent" if consistent else "volatile"
        
        # Determine trading recommendation
        if overall_change_pct > 2:
            recommendation = "STRONG BUY"
            rec_color = "green"
            action = "Consider buying with a target of ₹" + f"{last_prediction:.2f}"
        elif overall_change_pct > 0.5:
            recommendation = "BUY"
            rec_color = "green"
            action = "Consider gradual buying with a target of ₹" + f"{last_prediction:.2f}"
        elif overall_change_pct > -0.5:
            recommendation = "HOLD"
            rec_color = "blue"
            action = "Hold current positions and monitor the market"
        elif overall_change_pct > -2:
            recommendation = "SELL"
            rec_color = "red"
            action = "Consider selling to avoid potential losses"
        else:
            recommendation = "STRONG SELL"
            rec_color = "darkred"
            action = "Consider selling quickly to minimize losses"
        
        # Add analysis text in a box
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        analysis_text = f"""
        📊 STOCK PREDICTION ANALYSIS
        
        Model: {model_name}
        Time Frame: Next {len(future_prices)} Trading Days
        
        📈 PRICE FORECAST:
        • Current Price: ₹{current_price:.2f}
        • Predicted Final Price: ₹{last_prediction:.2f}
        • Expected Change: ₹{overall_change:.2f} ({overall_change_pct:.2f}%)
        
        🔍 TREND ANALYSIS:
        • Direction: {'Upward' if overall_change > 0 else 'Downward'}
        • Strength: {strength_desc.title()}
        • Pattern: {consistency_desc.title()}
        • Confidence: {self.models[model_name]['r2']:.4f} (R² score)
        
        💡 RECOMMENDATION: {recommendation}
        
        📝 SUGGESTED ACTION:
        {action}
        
        ⚠️ RISK ASSESSMENT:
        • Recent Volatility: ₹{self.df['Volatility'].iloc[-1]:.2f}
        • Market Conditions: {'Favorable' if self.df['RSI'].iloc[-1] < 70 and self.df['RSI'].iloc[-1] > 30 else 'Cautious'}
        
        Note: This prediction is based on historical patterns and may not account for unexpected market events.
        Always consider market news and your risk tolerance before making investment decisions.
        """
        
        plt.text(0.5, 0.5, analysis_text, ha='center', va='center', 
                 bbox=dict(boxstyle="round,pad=1", facecolor='lightgrey', alpha=0.4),
                 fontsize=12, transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
        
    def generate_trading_signals(self, model_name=None):
        """Generate detailed trading signals with actionable advice"""
        
        if model_name is None:
            comparison_df, model_name = self.evaluate_models()
            # If Linear Regression shows perfect R^2, it might be overfitting
            if model_name == 'Linear Regression' and comparison_df.iloc[0]['R²'] > 0.99:
                model_name = comparison_df.iloc[1]['Model']
        
        # Get predictions for the next 5 days
        future_prices = self.predict_future(model_name)
        
        if not future_prices:
            return
            
        current_price = self.df['Close'].iloc[-1]
        last_prediction = future_prices[-1]
        change_pct = ((last_prediction - current_price) / current_price) * 100
        
        # Get technical indicators for current state
        current_rsi = self.df['RSI'].iloc[-1]
        current_macd = self.df['MACD'].iloc[-1]
        current_macd_signal = self.df['MACD_signal'].iloc[-1]
        current_bb_width = self.df['BB_width'].iloc[-1]
        mean_bb_width = self.df['BB_width'].mean()
        
        # Calculate price momentum
        short_term_momentum = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[-5]) / self.df['Close'].iloc[-5] * 100
        medium_term_momentum = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[-20]) / self.df['Close'].iloc[-20] * 100
        
        print(f"\n{'='*50}")
        print("ADVANCED TRADING SIGNAL ANALYSIS")
        print(f"{'='*50}")
        
        # Determine signal
        if change_pct > 2:
            signal = "STRONG BUY"
            color = "🟢"
        elif change_pct > 0.5:
            signal = "BUY"
            color = "🟢"
        elif change_pct > -0.5:
            signal = "HOLD"
            color = "🟡"
        elif change_pct > -2:
            signal = "SELL"
            color = "🔴"
        else:
            signal = "STRONG SELL"
            color = "🔴"
        
        # Print main signal
        print(f"Primary Signal: {color} {signal}")
        print(f"Model: {model_name} (R² score: {self.models[model_name]['r2']:.4f})")
        print(f"Time Horizon: 5 Trading Days")
        print(f"Expected Return: {change_pct:.2f}%")
        
        # Risk assessment
        recent_volatility = self.df['Close'].tail(20).std()
        avg_volatility = self.df['Close'].std()
        relative_volatility = recent_volatility / avg_volatility
        
        if recent_volatility > 1000:
            risk_level = "HIGH"
            risk_color = "🔴"
        elif recent_volatility > 500:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "🟢"
        
        print(f"\n{'-'*50}")
        print("RISK ASSESSMENT")
        print(f"{'-'*50}")
        print(f"Risk Level: {risk_color} {risk_level}")
        print(f"Recent Volatility: ₹{recent_volatility:.2f}")
        print(f"Relative Volatility: {relative_volatility:.2f}x average")
        
        # Technical indicators analysis
        print(f"\n{'-'*50}")
        print("TECHNICAL INDICATORS")
        print(f"{'-'*50}")
        
        # RSI
        if current_rsi > 70:
            rsi_signal = "Overbought - Caution advised"
            rsi_icon = "⚠️"
        elif current_rsi < 30:
            rsi_signal = "Oversold - Potential buying opportunity"
            rsi_icon = "✅"
        else:
            rsi_signal = "Neutral"
            rsi_icon = "➖"
        
        print(f"RSI ({current_rsi:.2f}): {rsi_icon} {rsi_signal}")
        
        # MACD
        if current_macd > current_macd_signal:
            macd_signal = "Bullish - Positive momentum"
            macd_icon = "📈"
        else:
            macd_signal = "Bearish - Negative momentum"
            macd_icon = "📉"
        
        print(f"MACD: {macd_icon} {macd_signal}")
        
        # Bollinger Bands
        if current_bb_width > mean_bb_width * 1.5:
            bb_signal = "Wide - High volatility expected"
            bb_icon = "↔️"
        elif current_bb_width < mean_bb_width * 0.5:
            bb_signal = "Narrow - Breakout possible"
            bb_icon = "🔄"
        else:
            bb_signal = "Normal range"
            bb_icon = "➖"
            
        print(f"Bollinger Bands: {bb_icon} {bb_signal}")
        
        # Momentum
        print(f"5-Day Momentum: {short_term_momentum:.2f}%")
        print(f"20-Day Momentum: {medium_term_momentum:.2f}%")
        
        # Detailed recommendation
        print(f"\n{'-'*50}")
        print("DETAILED RECOMMENDATION")
        print(f"{'-'*50}")
        
        if signal == "STRONG BUY":
            print("✅ Strong buying opportunity indicated")
            print("Suggested Actions:")
            print("• Consider allocating 5-10% of available capital")
            print("• Set a target price of ₹" + f"{last_prediction * 1.02:.2f}")
            print("• Set a stop loss at ₹" + f"{current_price * 0.98:.2f}")
            if current_rsi > 60:
                print("• Exercise caution as RSI indicates potential overbought conditions")
        
        elif signal == "BUY":
            print("✅ Moderate buying opportunity indicated")
            print("Suggested Actions:")
            print("• Consider phased buying (2-5% of available capital)")
            print("• Set a target price of ₹" + f"{last_prediction * 1.01:.2f}")
            print("• Set a stop loss at ₹" + f"{current_price * 0.97:.2f}")
        
        elif signal == "HOLD":
            print("⏺️ Hold current positions")
            print("Suggested Actions:")
            print("• Maintain current positions")
            print("• No new buying recommended")
            print("• No selling recommended unless you need liquidity")
            print("• Review again in 3-5 trading days")
        
        elif signal == "SELL":
            print("❎ Consider reducing positions")
            print("Suggested Actions:")
            print("• Consider selling 30-50% of holdings")
            print("• Set a target re-entry price of ₹" + f"{current_price * 0.97:.2f}")
        
        else:  # STRONG SELL
            print("❎ Strong sell signal detected")
            print("Suggested Actions:")
            print("• Consider selling 70-100% of holdings")
            print("• Set a target re-entry price of ₹" + f"{current_price * 0.95:.2f}")
            print("• Re-evaluate market conditions after the predicted period")
        
        print(f"\n{'-'*50}")
        print("IMPORTANT CAVEATS")
        print(f"{'-'*50}")
        print("• This analysis is based on historical patterns and may not predict unexpected events")
        print("• Always diversify your investments and avoid committing too much capital to a single trade")
        print("• Consider consulting a financial advisor for personalized investment advice")
        print(f"{'-'*50}")
    
    def create_dashboard(self):
        """Create a comprehensive dashboard"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Price Movement
        plt.subplot(3, 3, 1)
        plt.plot(self.df['Date'], self.df['Close'], linewidth=2)
        plt.title('Stock Price Movement', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Volume
        plt.subplot(3, 3, 2)
        plt.plot(self.df['Date'], self.df['Shares Traded'], color='orange', linewidth=2)
        plt.title('Trading Volume', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Shares Traded')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Moving Averages
        plt.subplot(3, 3, 3)
        plt.plot(self.df['Date'], self.df['Close'], label='Close', linewidth=2)
        plt.plot(self.df['Date'], self.df['SMA_20'], label='SMA 20', linewidth=2)
        plt.plot(self.df['Date'], self.df['EMA_12'], label='EMA 12', linewidth=2)
        plt.title('Moving Averages', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. RSI
        plt.subplot(3, 3, 4)
        plt.plot(self.df['Date'], self.df['RSI'], color='purple', linewidth=2)
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        plt.title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. MACD
        plt.subplot(3, 3, 5)
        plt.plot(self.df['Date'], self.df['MACD'], label='MACD', linewidth=2)
        plt.plot(self.df['Date'], self.df['MACD_signal'], label='Signal', linewidth=2)
        plt.title('MACD', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Bollinger Bands
        plt.subplot(3, 3, 6)
        plt.plot(self.df['Date'], self.df['Close'], label='Close', linewidth=2)
        plt.plot(self.df['Date'], self.df['BB_upper'], label='Upper Band', alpha=0.7)
        plt.plot(self.df['Date'], self.df['BB_lower'], label='Lower Band', alpha=0.7)
        plt.fill_between(self.df['Date'], self.df['BB_upper'], self.df['BB_lower'], alpha=0.1)
        plt.title('Bollinger Bands', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Volatility
        plt.subplot(3, 3, 7)
        plt.plot(self.df['Date'], self.df['Volatility'], color='red', linewidth=2)
        plt.title('Price Volatility', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. Price vs Volume
        plt.subplot(3, 3, 8)
        plt.scatter(self.df['Shares Traded'], self.df['Close'], alpha=0.6)
        plt.title('Price vs Volume Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Shares Traded')
        plt.ylabel('Price (₹)')
        plt.grid(True, alpha=0.3)
        
        # 9. Daily Returns Distribution
        plt.subplot(3, 3, 9)
        daily_returns = self.df['Close'].pct_change().dropna() * 100
        plt.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Daily Returns Distribution (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("STOCK ANALYSIS SUMMARY")
        print("="*60)
        print(f"Current Price: ₹{self.df['Close'].iloc[-1]:.2f}")
        print(f"52-Week High: ₹{self.df['Close'].max():.2f}")
        print(f"52-Week Low: ₹{self.df['Close'].min():.2f}")
        print(f"Average Volume: {self.df['Shares Traded'].mean():,.0f}")
        print(f"Current RSI: {self.df['RSI'].iloc[-1]:.2f}")
        print(f"Average Daily Return: {daily_returns.mean():.2f}%")
        print(f"Volatility (Std Dev): {daily_returns.std():.2f}%")

def main():
    """Main function to run the stock prediction system"""
    
    print("🚀 ADVANCED STOCK PREDICTION SYSTEM")
    print("="*50)
    
    # Initialize the predictor with all CSV files
    csv_files = [
        "NIFTY_50_24-25.csv",
        "NIFTY_50_23-24.csv",
        "NIFTY_50_22-23.csv",
        "NIFTY_50_21-22.csv"
    ]
    
    print(f"Using {len(csv_files)} CSV files for prediction:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Create the predictor with all CSV files
    predictor = StockPredictor(csv_files)
    
    # Load and preprocess data
    if not predictor.load_and_preprocess_data():
        return
    
    # Create technical indicators
    predictor.create_technical_indicators()
    
    # Train models
    print("\n📊 Training machine learning models...")
    results = predictor.train_models()
    
    # Evaluate models
    print("\n📈 Evaluating model performance...")
    comparison_df, best_model = predictor.evaluate_models()
    
    # Generate predictions and trading signals
    print(f"\n🎯 Generating predictions using {best_model}...")
    predictor.predict_future(best_model)
    predictor.generate_trading_signals(best_model)
    
    # Show visualizations
    print("\n📊 Creating visualizations...")
    predictor.create_dashboard()
    predictor.plot_predictions(best_model)
    predictor.get_feature_importance('Random Forest')
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()