# PADSI API
This is a Flask application that predicts trading behavior using machine learning models. It fetches trading data from Binance exchange and applies various models to make predictions.

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the Binance API credentials:
Open the config.py file and provide your Binance API key and secret.

4. Start the Flask application:

```bash
python app.py
```
The application will run on http://localhost:5000.

## Endpoints
The application exposes the following endpoints:

* /getKNN25/<token>: Returns the prediction from the KNN model with a time frequency of 25 seconds for the specified token.
* /getKNN15/<token>: Returns the prediction from the KNN model with a time frequency of 15 seconds for the specified token.
* /getKNN5/<token>: Returns the prediction from the KNN model with a time frequency of 5 seconds for the specified token.
* /getSVM25/<token>: Returns the prediction from the SVM model with a time frequency of 25 seconds for the specified token.
* /getSVM15/<token>: Returns the prediction from the SVM model with a time frequency of 15 seconds for the specified token.
* /getSVM5/<token>: Returns the prediction from the SVM model with a time frequency of 5 seconds for the specified token.
* /getDNN25/<token>: Returns the prediction from the DNN model with a time frequency of 25 seconds for the specified token.
* /getDNN15/<token>: Returns the prediction from the DNN model with a time frequency of 15 seconds for the specified token.
* /getDNN5/<token>: Returns the prediction from the DNN model with a time frequency of 5 seconds for the specified token.

Replace <token> with the trading symbol or token you want to make predictions for.

## Usage
To use the application, send HTTP GET requests to the desired endpoint using a tool like cURL or a web browser. For example:

bash
Copy code
curl http://localhost:5000/getKNN25/BTCUSDT
This will return the prediction from the KNN model with a time frequency of 25 seconds for the BTCUSDT trading pair.

## Citation

The trading prediction algorithm in this project is inspired by the research conducted by M. La Morgia, A. Mei, F. Sassi, and J. Stefa, who developed a real-time detection method for cryptocurrency market manipulations using pump and dumps techniques 

[M. La Morgia, A. Mei, F. Sassi, J. Stefa. "Pump and Dumps in the Bitcoin Era: Real Time Detection of Cryptocurrency Market Manipulations." In 2020 29th International Conference on Computer Communications and Networks (ICCCN), pages 1-9, 2020.](https://doi.org/10.1109/ICCCN49398.2020.9209660)
