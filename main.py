from dataset import preprocessing
from model import LinearRegression
from predict import predict
from train import train

hyperparams = {
    'lr': 0.01,
    'epochs': 500,
    'l1_lambda': 0.01
}

data_path = 'data/data.csv'
predict_data_path = 'data/data_predict.csv'

if __name__ == '__main__':
    print("\n--------Training--------")
    X_train, y_train, preprocessor = preprocessing(path=data_path)

    in_size = X_train.shape[1]
    model = LinearRegression(in_size=in_size)

    trained_model = train(model, X_train, y_train, **hyperparams)

    # print("Saving model to disk...")
    # if not os.path.exists("model"):
    #     os.makedirs("model")
    # torch.save(trained_model.state_dict(), 'model/biscuit_model.pth')
    # joblib.dump(preprocessor, 'model/biscuit_preprocessor.joblib')

    print("\n--------Prediction--------")
    name, result, target = predict(model, preprocessor, predict_data_path)

    print("\n--------Results--------")
    for name, score, target in zip(name, result, target):
        print(f"{name}: {score:.2f} (expected: {target:.2f})")