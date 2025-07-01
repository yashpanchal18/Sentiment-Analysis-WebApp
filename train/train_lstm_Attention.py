import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Attention,
    Permute,
    Multiply,
    RepeatVector,
    Activation,
    Concatenate,
    Flatten  
)
from tensorflow.keras.optimizers import Adam
import joblib
from tensorflow.keras.layers import GlobalAveragePooling1D
from data_loader import load_dataset
from preprocess import encode_reviews_bilstm

def build_lstm_with_attention(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    
    # Attention mechanism
    attention_weights = Dense(1, activation='tanh')(lstm_out)
    attention_weights = Flatten()(attention_weights)
    attention_weights = Activation('softmax')(attention_weights)
    attention_weights = RepeatVector(64)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)

    weighted = Multiply()([lstm_out, attention_weights])
    representation = GlobalAveragePooling1D()(weighted)

    
    x = Dropout(0.3)(representation)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

def train_lstm_attention_model():
    print("🔄 Loading dataset...")
    reviews, labels, label_encoder = load_dataset(r"C:\Users\Lenovo\Desktop\XAI\SentX\data\reviews_dataset_small.csv")
    print("🧠 Encoding reviews into FastText sequences...")
    encoded_reviews = encode_reviews_bilstm(reviews, max_len=30)

    # One-hot encode labels for categorical crossentropy
    num_classes = 3
    labels = to_categorical(labels, num_classes=num_classes)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_reviews, labels, test_size=0.2, random_state=42
    )

    print("⚙️ Building LSTM model with Attention...")
    model = build_lstm_with_attention((30, 300), num_classes)

    print("🚀 Training...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1,
    )

    print("💾 Saving model and label encoder...")
    model.save(r"C:\Users\Lenovo\Desktop\XAI\SentX\models\lstm_attention_model.h5")
    joblib.dump(label_encoder, r"C:\Users\Lenovo\Desktop\XAI\SentX\models\lstm_attention_label_encoder.pkl")
    print("✅ Done.")

if __name__ == "__main__":
    train_lstm_attention_model()
