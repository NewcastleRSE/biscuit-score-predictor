# Biscuit Score Prediction Model

A machine learning tool to predict scores for biscuits based on their nutritional information and other features (colour, texture...). 

## Installation
1.  **Clone the repository:**
    ```bash
    git clone git@github.com:NewcastleRSE/biscuit_score_predictor.git
    cd biscuit_score_predictor
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Model

To train the model, run the main script. 
This will read the `data.csv` file.
```
python main.py
```

The script also make the prediction using the information in `data_predict.csv`. Example results:
```
Fox's Crunch Cream Style : 3.76 (expected: 3.01)
Biscoff Style: 5.94 (expected: 7.45)
```
