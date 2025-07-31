#Loads processed training data, builds and trains a Keras neural network with early stopping, saves the trained model.
# train.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')

    model = build_model(X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        callbacks=[early_stop],
        verbose=1
    )

    model.save('models/churn_model.h5')
    print("Model training complete and saved.")

if __name__ == '__main__':
    main()
