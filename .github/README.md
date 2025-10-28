# Hidden Markov Model (HMM) POS Tagger

## Aim

The aim of this project is the **Implementation of a Part-of-Speech (POS) Tagger** using the **Hidden Markov Model (HMM)** methodology. This includes using the NLTK Treebank corpus for training and creating an interactive web interface using Streamlit for demonstration.

## Objectives

1. To build a **supervised HMM-based POS Tagger** using the `nltk.tag.hmm` library.

2. To train and evaluate the model using a defined training split of the **NLTK Treebank corpus**.

3. To create a **Streamlit web application** for interactive sentence tagging.

4. To display the POS tags in a **human-readable format**, expanding abbreviations (e.g., NN → Noun).

5. To implement **model persistence** using the `dill` library to save and load the trained tagger, avoiding redundant training.

## System Requirements

| Category              | Requirement                                       |
| :-------------------- | :------------------------------------------------ |
| Programming Language  | Python 3.10+                                      |
| Core NLP Library      | `nltk` (Natural Language Toolkit)                 |
| Web App Framework     | `streamlit`                                       |
| Serialization Library | `dill` (for model persistence)                    |
| NLTK Corpora          | `treebank`, `punkt`, `averaged_perceptron_tagger` |

## Methodology and Theory

### Data Source

The **Penn Treebank Corpus**, available through NLTK, serves as the labeled training data. It provides sentences already tokenized and tagged with the standard Penn Treebank tag set (e.g., NN, VBZ, JJ).

### Hidden Markov Model (HMM)

An HMM is a probabilistic sequence model that is effective for tasks like POS tagging. It relies on the Viterbi algorithm to find the single best sequence of hidden states (POS tags) given a sequence of observations (words).

The model is defined by two primary probability tables derived from the training data:

1. **Transition Probabilities** $P(\text{tag}_i \mid \text{tag}_{i-1})$: The probability of moving to a specific tag ($\text{tag}_i$) given the previous tag ($\text{tag}_{i-1}$).

2. **Emission Probabilities** $P(\text{word}_i \mid \text{tag}_i)$: The probability of observing a specific word ($\text{word}_i$) given a specific tag ($\text{tag}_i$).

### Training and Persistence

The `nltk.tag.hmm.HiddenMarkovModelTrainer` class is used for supervised training. Once trained, the model is serialized (pickled) using `dill` and saved to `hmm_pos_tagger.pkl`.

The application logic ensures the model is loaded from this file on startup, making subsequent runs significantly faster. Retraining only occurs if the user explicitly requests it or if the model file is not found.

## Implementation Details

The system is organized into two main files: `hmm_tagger.py` (core logic) and `app.py` (Streamlit interface).

### `hmm_tagger.py`

Handles model training, loading, tagging, and formatting.

**Key functions:**

-   `ensure_nltk_data()`: Ensures all necessary NLTK corpora are downloaded.

-   `train_or_load_tagger()`: Manages the efficient loading of the tagger from disk or initiating training if necessary.

-   `tag_sentence()`: Takes a string, tokenizes it, and applies the HMM model to return the list of (word, tag) tuples.

-   `format_tagged_output()`: This crucial function expands the abbreviated POS tags into descriptive labels (e.g., **JJS** to "Adjective, superlative" and **NN** to "Noun, singular or mass") and formats the output for display.

### `app.py`

Implements the Streamlit UI for an interactive experience.

-   Sets up the application layout and a session state for the tagger.

-   Allows the user to select the **training size** (number of sentences from the Treebank).

-   Includes a button to explicitly **"Retrain Model"**.

-   Takes user input (the sentence to be tagged).

-   Calls the functions from `hmm_tagger.py` and displays the formatted results in an easy-to-read code block.

## Example Output

**Input Sentence:**
`The quick brown fox jumps over the lazy dog.`

**Output (Formatted by `format_tagged_output`):**

The → Determiner  
quick → Adjective  
brown → Adjective  
fox → Noun, singular or mass  
jumps → Verb, 3rd person singular present  
over → Preposition or subordinating conjunction  
the → Determiner  
lazy → Adjective  
dog → Noun, singular or mass

## Conclusion

This project successfully demonstrates the implementation of a **Hidden Markov Model-based POS Tagger** using the NLTK library. By incorporating model persistence with `dill` and an intuitive Streamlit interface, the system provides an effective and efficient tool for understanding sequence modeling and POS tagging in NLP. The custom formatting successfully bridges the technical output of the HMM with human-readable grammar descriptions.
