# Bitcoin Price Prediction using LSTM

This project demonstrates how to use an LSTM (Long Short-Term Memory) neural network to predict the closing price of Bitcoin based on historical data. It includes data preprocessing, sequence generation, model training, and visualization of the results.

## 📊 Dataset

- Source: `BTC-USD (1).csv`
- Features used:
  - Open
  - High
  - Low
  - Close (Target)
  - Volume
- The dataset is cleaned by removing missing values and sorted by date.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn (MinMaxScaler)

## 🔁 Data Preprocessing

- The data is normalized using `MinMaxScaler` to improve training performance.
- A sequence length of 60 time steps is used to train the model (each input sample consists of 60 previous days).
- Data is split into 80% training and 20% testing.

## 🧠 Model Architecture

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 5)))
model.add(LSTM(50))
model.add(Dense(1))
