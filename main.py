import streamlit as st
from inference import generate_caption


def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and generate a caption using the trained model.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image is not None:
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        model_path = "Model/model.keras"
        tokenizer_path = "Model/tokenizer.pkl"
        feature_extractor_path = "Model/feature_extractor.keras"

        caption = generate_caption(uploaded_image, model_path, tokenizer_path, feature_extractor_path)
        # st.write("Generated Caption:", caption)
        st.image(uploaded_image, caption=caption, width=500)


if __name__ == "__main__":
    main()
