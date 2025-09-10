import re
from collections import Counter
import os
import math

def train_language_model(corpus_path, output_dir):
    """
    Trains a simple n-gram language model and saves it in ARPA format.
    """
    print(f"Reading corpus from: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()

    # Simple tokenization
    print("Tokenizing text...")
    sentences = text.split('\n')
    words = [word for sentence in sentences for word in re.findall(r'\b\w+\b', sentence)]

    # Count n-grams
    print("Counting n-grams...")
    unigrams = Counter(words)
    bigrams = Counter(zip(words, words[1:]))
    trigrams = Counter(zip(words, words[1:], words[2:]))

    V = len(unigrams)
    N = len(words)

    output_path = os.path.join(output_dir, "lm.arpa")
    print(f"Writing ARPA file to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\data\
")
        f.write(f"ngram 1={len(unigrams)}\n")
        f.write(f"ngram 2={len(bigrams)}\n")
        f.write(f"ngram 3={len(trigrams)}\n")
        
        # Unigrams
        f.write("\n\1-grams:\n")
        for word, count in unigrams.items():
            prob = (count + 1) / (N + V)
            log_prob = math.log10(prob)
            # Using a dummy backoff weight for now
            f.write(f"{log_prob:.4f}\t{word}\t-99.0000\n")

        # Bigrams
        f.write("\n\2-grams:\n")
        for bigram, count in bigrams.items():
            unigram_count = unigrams[bigram[0]]
            prob = (count + 1) / (unigram_count + V)
            log_prob = math.log10(prob)
            # Using a dummy backoff weight for now
            f.write(f"{log_prob:.4f}\t{' '.join(bigram)}\t-99.0000\n")

        # Trigrams
        f.write("\n\3-grams:\n")
        for trigram, count in trigrams.items():
            bigram_count = bigrams[trigram[:2]]
            prob = (count + 1) / (bigram_count + V)
            log_prob = math.log10(prob)
            f.write(f"{log_prob:.4f}\t{' '.join(trigram)}\n")
            
        f.write("\n\end\
")
    
    print("Finished writing ARPA file with probabilities.")

if __name__ == '__main__':
    corpus_file = 'data/corpus.txt'
    output_dir = 'data/lm'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_language_model(corpus_file, output_dir)