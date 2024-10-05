# AI-Driven Cryptocurrency Price Forecasting

## Description
My project uses AI techniques, specifically linear regression and LSTM neural networks, to predict short-term cryptocurrency prices. Historical data from popular exchanges is analyzed to forecast the prices of Bitcoin, Ethereum, BNB, and Dogecoin for the last 50 days. Note: The accuracy of these predictions is measured using MSE to assess the effectiveness of AI in financial forecasting.

## Features
- Predict short-term cryptocurrency prices using linear regression and LSTM neural networks.
- Analyze historical price data from popular exchanges.
- Evaluate model performance using Mean Squared Error (MSE).


## Usage
1. Run the preprocessing and data preparation steps found in the `Cryptocurrencies Project Machine Learning.ipynb` notebook.
2. Train the linear regression model using the `Lineal_Regresion_Model.ipynb` notebook.
3. Train and evaluate the LSTM model using the `Neural Networks Price Prediction.ipynb` notebook.
4. Compare the performance of both models and analyze the results.

# Neural Networks Price Prediction.ipynb

1. **Run the Preprocessing and Data Preparation:**
   - Load and clean the historical Bitcoin data.
   - Filter the data between specific dates and convert the 'Close' prices columns to a Dataframe.
   - Split the data into training and test sets.

2. **Scale the Data:**
   - Use `MinMaxScaler` to scale the data between 0 and 1.
   ```python
   scaler = MinMaxScaler(feature_range=(0, 1))
   dataset_train = scaler.fit_transform(dataset_train)
   dataset_test = scaler.transform(dataset_test)
   ```

3. **Create Datasets:**
   - Create sequences of 50 days to predict the next day’s price.
   ```python
   def create_my_dataset(df):
       x = []
       y = []
       for i in range(50, df.shape[0]):
           x.append(df[i-50:i, 0])
           y.append(df[i, 0])
       x = np.array(x)
       y = np.array(y)
       return x, y

   x_train, y_train = create_my_dataset(dataset_train)
   x_test, y_test = create_my_dataset(dataset_test)
   ```

4. **Build the Model:**
   - Build and compile the LSTM model.
   ```python
   model = Sequential()
   model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
   model.add(Dropout(0.2))
   model.add(LSTM(units=96, return_sequences=True))
   model.add(Dropout(0.2))
   model.add(LSTM(units=96))
   model.add(Dropout(0.2))
   model.add(Dense(units=1))

   model.compile(loss='mean_squared_error', optimizer='adam')
   ```

5. **Train and Save the Model:**
   - Train the model and save it for future use.
   ```python
   if not os.path.exists("Prices_Prediction.h5"):
       model.fit(x_train, y_train, epochs=50, batch_size=32)
       model.save("Prices_Prediction.h5")

   model = load_model("Prices_Prediction.h5")
   ```

6. **Make Predictions and Visualize:**
   - Predict prices on the test set and plot the results.
   ```python
   predictions = model.predict(x_test)
   predictions = scaler.inverse_transform(predictions)

   plt.figure(figsize=(8, 4))
   plt.plot(df, color='red', label='Original Stock Price')
   plt.plot(range(len(y_train) + 50, len(y_train) + 50 + len(predictions)), predictions, color='blue', label='Predictions')
   plt.legend()
   plt.show()
   ```

# Lineal_Regresion_Model.ipynb

### Usage
1. **Run the Preprocessing and Data Preparation:**
   - Load and clean the historical Bitcoin data.
   - Filter the data between specific dates and convert the 'Close' price column to a numpy array.
   - Split the data into training and test sets.

2. **Scale the Data:**
   - Use `MinMaxScaler` to scale the data between 0 and 1.
   ```python
   scaler = MinMaxScaler(feature_range=(0, 1))
   dataset_train = scaler.fit_transform(dataset_train)
   dataset_test = scaler.transform(dataset_test)
   ```

3. **Create Datasets:**
   - Create sequences of 50 days to predict the next day’s price.
   ```python
   def create_my_dataset(df):
       x = []
       y = []
       for i in range(50, df.shape[0]):
           x.append(df[i-50:i, 0])
           y.append(df[i, 0])
       x = np.array(x)
       y = np.array(y)
       return x, y

   x_train, y_train = create_my_dataset(dataset_train)
   x_test, y_test = create_my_dataset(dataset_test)
   ```

4. **Build the Model:**
   - Build and compile the LSTM model.
   ```python
   model = Sequential()
   model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
   model.add(Dropout(0.2))
   model.add(LSTM(units=96, return_sequences=True))
   model.add(Dropout(0.2))
   model.add(LSTM(units=96))
   model.add(Dropout(0.2))
   model.add(Dense(units=1))

   model.compile(loss='mean_squared_error', optimizer='adam')
   ```

5. **Train and Save the Model:**
   - Train the model and save it for future use.
   ```python
   if not os.path.exists("Prices_Prediction.h5"):
       model.fit(x_train, y_train, epochs=50, batch_size=32)
       model.save("Prices_Prediction.h5")

   model = load_model("Prices_Prediction.h5")
   ```

6. **Make Predictions and Visualize:**
   - Predict prices on the test set and plot the results.
   ```python
   predictions = model.predict(x_test)
   predictions = scaler.inverse_transform(predictions)

   plt.figure(figsize=(8, 4))
   plt.plot(df, color='red', label='Original Stock Price')
   plt.plot(range(len(y_train) + 50, len(y_train) + 50 + len(predictions)), predictions, color='blue', label='Predictions')
   plt.legend()
   plt.show()
   ```

---

### Usage for Linear Regression Model
1. **Data Preparation:**
   - Load and clean the historical Bitcoin data.
   - Filter the data between specific dates.
   ```python
   df = pd.read_csv('path_to/Bitcoin_data.csv')
   df['Date'] = pd.to_datetime(df['Date'])
   df = df[(df['Date'] >= '2017-08-01') & (df['Date'] <= '2024-07-09')]
   ```

2. **Feature Selection and Target Definition:**
   - Select features and define the target for the next day's closing price.
   ```python
   df['Next Close'] = df['Close'].shift(-1)
   df = df.dropna()
   X = df[['Close', 'Volume', 'Market Cap']]
   y = df['Next Close']
   ```

3. **Remove Outliers (Optional):**
   - Optionally, remove outliers using IQR or Z-score methods.
   ```python
   def remove_outliers_iqr(X, y):
       Q1 = X.quantile(0.25)
       Q3 = X.quantile(0.75)
       IQR = Q3 - Q1
       condition = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
       mask = ~condition.any(axis=1)
       return X[mask], y[mask]

   def remove_outliers_zscore(X, y, threshold=3):
       zs = np.abs(zscore(X))
       mask = (zs < threshold).all(axis=1)
       return X[mask], y[mask]
   ```

4. **Scale Data and Train the Model:**
   - Scale the data and train the linear regression model.
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   X_train, X_test = X_scaled[:-50], X_scaled[-50:]
   y_train, y_test = y[:-50], y[-50:]
   model = LinearRegression()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

5. **Evaluate the Model:**
   - Evaluate the model using metrics like MSE, MAE, RMSE, and R2.
   ```python
   mse = mean_squared_error(y_test, predictions)
   mae = mean_absolute_error(y_test, predictions)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_test, predictions)
   print({'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
   ```

6. **Plot the Results:**
   - Plot the actual vs. predicted prices.
   ```python
   plt.figure(figsize=(15, 7))
   plt.plot(range(len(y_test)), y_test, label='Actual Price', color='black')
   plt.plot(range(len(predictions)), predictions, label='Predicted Price', linestyle='--')
   plt.title('Comparison of Actual and Predicted Prices')
   plt.xlabel('Days')
   plt.ylabel('Bitcoin Price (USD)')
   plt.legend()
   plt.show()
   ```
