"""Streamlit dashboard for the CIFAR-10 CNN image classifier.

Run with: streamlit run app.py
The app looks for cifar10_cnn_final.keras or best_cifar10_cnn.keras in this folder.
If neither exists, train the model from the sidebar (this may take a while).
"""
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🧠", layout="wide")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
MODEL_CANDIDATES = [Path("cifar10_cnn_final.keras"), Path("best_cifar10_cnn.keras")]


def build_model(tf):
    """Build the same CNN architecture as the notebook."""
    layers = tf.keras.layers
    model = tf.keras.Sequential(name="CIFAR10_CNN")
    model.add(layers.Input(shape=(32, 32, 3)))
    for block, filters in enumerate((32, 64, 128), start=1):
        for conv in (1, 2):
            model.add(layers.Conv2D(filters, (3, 3), padding="same", name=f"conv{block}_{conv}"))
            model.add(layers.BatchNormalization(name=f"bn{block}_{conv}"))
            model.add(layers.Activation("relu", name=f"relu{block}_{conv}"))
        model.add(layers.MaxPooling2D((2, 2), name=f"pool{block}"))
        model.add(layers.Dropout(0.25, name=f"drop{block}"))
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(256, name="dense1"))
    model.add(layers.BatchNormalization(name="bn_dense"))
    model.add(layers.Activation("relu", name="relu_dense"))
    model.add(layers.Dropout(0.5, name="drop_dense"))
    model.add(layers.Dense(10, activation="softmax", name="output"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource(show_spinner=False)
def load_existing_model(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def get_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


st.title("🧠 CIFAR-10 Image Classification")
st.caption("A Convolutional Neural Network (CNN) that classifies small images into 10 object categories.")

with st.sidebar:
    st.header("Model")
    st.write("Upload a picture or explore the CIFAR-10 dataset.")
    model_path = get_model_path()
    if model_path:
        st.success(f"Model found: `{model_path.name}`")
        try:
            model = load_existing_model(str(model_path))
        except Exception as exc:
            model = None
            st.error(f"Could not load model: {exc}")
    else:
        model = None
        st.warning("No trained model file found.")
        st.caption("Train a model below, or place `cifar10_cnn_final.keras` in this folder.")

    with st.expander("Train a new model"):
        epochs = st.slider("Training epochs", min_value=1, max_value=25, value=5)
        batch_size = st.select_slider("Batch size", options=[32, 64, 128], value=64)
        if st.button("Train CNN", type="primary", use_container_width=True):
            try:
                import tensorflow as tf
                from tensorflow.keras.preprocessing.image import ImageDataGenerator
                with st.spinner("Loading CIFAR-10 and training the CNN. This can take a while..."):
                    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
                    y_train, y_test = y_train.flatten(), y_test.flatten()
                    x_train = x_train.astype("float32") / 255.0
                    x_test = x_test.astype("float32") / 255.0
                    tf.keras.utils.set_random_seed(42)
                    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                        height_shift_range=0.1, horizontal_flip=True, zoom_range=0.1)
                    fresh_model = build_model(tf)
                    callbacks = [
                        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
                    ]
                    progress = st.progress(0, text="Preparing training...")
                    status = st.empty()
                    class StreamlitProgress(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress.progress((epoch + 1) / epochs, text=f"Completed epoch {epoch + 1} of {epochs}")
                            logs = logs or {}
                            status.write(f"Epoch {epoch + 1}: loss={logs.get('loss', 0):.4f}, accuracy={logs.get('accuracy', 0):.3f}, validation accuracy={logs.get('val_accuracy', 0):.3f}")
                    history = fresh_model.fit(datagen.flow(x_train, y_train, batch_size=batch_size, seed=42),
                        epochs=epochs, steps_per_epoch=max(1, len(x_train) // batch_size),
                        validation_data=(x_test, y_test), callbacks=callbacks + [StreamlitProgress()], verbose=0)
                    fresh_model.save("cifar10_cnn_final.keras")
                st.success("Training complete! Model saved as `cifar10_cnn_final.keras`.")
                st.info("Refresh the page to load the saved model.")
                st.session_state["last_history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            except Exception as exc:
                st.error(f"Training failed: {exc}")


tab_overview, tab_predict, tab_dataset, tab_metrics = st.tabs(["Overview", "Predict an Image", "Explore Dataset", "Model Details"])

with tab_overview:
    st.subheader("Project overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Image classes", "10")
    m2.metric("Image size", "32 × 32 px")
    m3.metric("Training images", "50,000")
    st.markdown("""
    This app demonstrates a CNN trained on the **CIFAR-10** dataset. The network uses convolutional layers to learn visual features, batch normalization and ReLU activations, max pooling to reduce spatial dimensions, and dropout to help reduce overfitting.

    **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

    **How to use:**
    1. Put a trained `.keras` model beside `app.py`, or train one from the sidebar.
    2. Open **Predict an Image** and upload a picture.
    3. Review the predicted class and class confidence scores.

    *The model was designed for 32×32 CIFAR-10-style images. Predictions on unrelated photos may be unreliable.*
    """)
    if "last_history" in st.session_state:
        st.subheader("Latest training history")
        hist = st.session_state["last_history"]
        if "accuracy" in hist:
            st.line_chart({k: hist[k] for k in ["accuracy", "val_accuracy"] if k in hist})

with tab_predict:
    st.subheader("Upload an image to classify")
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Uploaded image", use_container_width=True)
            if model is None:
                with right:
                    st.warning("A trained model is required for prediction. Use **Train CNN** in the sidebar or add a `.keras` model file.")
            else:
                resized = image.resize((32, 32))
                arr = np.asarray(resized, dtype=np.float32) / 255.0
                probabilities = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                best_idx = int(np.argmax(probabilities))
                with right:
                    st.success(f"Prediction: **{CLASS_NAMES[best_idx].title()}**")
                    st.metric("Model confidence", f"{probabilities[best_idx] * 100:.1f}%")
                    st.caption("The image is resized to 32 × 32 pixels before prediction.")
                    ranking = np.argsort(probabilities)[::-1]
                    st.markdown("**Class probabilities**")
                    st.bar_chart({CLASS_NAMES[i]: float(probabilities[i]) for i in ranking})
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
    else:
        st.info("Upload a JPG, PNG, or WEBP image to get a prediction.")

with tab_dataset:
    st.subheader("Explore CIFAR-10")
    st.write("The dataset contains 60,000 colour images: 50,000 training images and 10,000 test images.")
    if st.button("Load example images", key="load_examples"):
        try:
            import tensorflow as tf
            with st.spinner("Loading CIFAR-10 dataset..."):
                (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
            y_train = y_train.flatten()
            fig, axes = plt.subplots(2, 5, figsize=(12, 5))
            for class_id, ax in enumerate(axes.flat):
                idx = int(np.where(y_train == class_id)[0][0])
                ax.imshow(x_train[idx])
                ax.set_title(CLASS_NAMES[class_id].title())
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Could not load dataset: {exc}")

with tab_metrics:
    st.subheader("CNN architecture")
    st.markdown("""
    - **Input:** 32 × 32 RGB image
    - **Feature extraction:** three convolutional blocks with 32, 64, and 128 filters
    - **Layers:** Conv2D, Batch Normalization, ReLU, Max Pooling, and Dropout
    - **Classifier:** Flatten → Dense (256) → Dropout → Softmax (10 classes)
    - **Optimizer:** Adam
    - **Loss:** Sparse categorical cross-entropy
    """)
    if model is not None:
        st.subheader("Loaded model summary")
        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        st.code("\n".join(summary_lines), language="text")
    else:
        st.info("Load or train a model to display its summary here.")

st.divider()
st.caption("Built with Streamlit, TensorFlow/Keras, NumPy, Pillow, and Matplotlib.")
