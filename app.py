import streamlit as st
from hmm_tagger import train_or_load_tagger, tag_sentence, format_tagged_output

st.set_page_config(page_title="HMM POS Tagger", page_icon="🧠", layout="centered")

st.title("Hidden Markov Model POS Tagger")
st.markdown(
    "Enter a sentence below, and the HMM model will predict each word’s **Part of Speech (POS)** tag."
)

train_size = st.slider(
    "Select training size (from NLTK Treebank):", 500, 5000, 2000, step=500
)
force_retrain = st.checkbox("Retrain Model", value=False)

if "tagger" not in st.session_state or force_retrain:
    with st.spinner("Training HMM POS Tagger... this may take a minute"):
        st.session_state["tagger"] = train_or_load_tagger(
            train_size=train_size, force_retrain=force_retrain
        )
    st.success("Model ready!")

sentence = st.text_area(
    "Enter your sentence here:", "The quick brown fox jumps over the lazy dog."
)

if st.button("Tag Sentence"):
    tagger = st.session_state["tagger"]
    tagged = tag_sentence(tagger, sentence)
    st.subheader("Tagged Output:")
    st.code(format_tagged_output(tagged), language="text")
