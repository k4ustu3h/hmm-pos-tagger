import nltk
import dill
import os
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer

MODEL_PATH = "hmm_pos_tagger.pkl"


def ensure_nltk_data():
    resources = ["treebank", "punkt", "averaged_perceptron_tagger", "wordnet"]
    for res in resources:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def train_or_load_tagger(train_size=2000, force_retrain=False):
    ensure_nltk_data()

    if not force_retrain and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                tagger = dill.load(f)
                print("Loaded saved HMM tagger.")
                return tagger
        except Exception as e:
            print(f"Failed to load existing model: {e}. Retraining...")

    print("Training new HMM tagger...")

    tagged_sents = treebank.tagged_sents()[:train_size]
    trainer = HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(tagged_sents)

    with open(MODEL_PATH, "wb") as f:
        dill.dump(tagger, f)

    print("HMM tagger trained and saved successfully.")
    return tagger


def tag_sentence(tagger, sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = tagger.tag(tokens)
    return tagged


POS_EXPANSIONS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition/Subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular or mass",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund/present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
}


def format_tagged_output(tagged):
    lines = []
    for word, tag in tagged:
        expanded = POS_EXPANSIONS.get(tag, tag)
        lines.append(f"{word:<15} → {expanded}")
    return "\n".join(lines)
