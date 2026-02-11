import pandas as pd
from collections import Counter

# Step 1: Preprocess the kreuzeandtatoeba.txt file to create a frequency-sorted list of unique words
def preprocess_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        word_freq = Counter(words)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words

# Preprocess the text file
sorted_words = preprocess_text_file('kreuzeandtatoeba.txt')

# Save the frequency list to a file
with open('word_frequencies.txt', 'w', encoding='utf-8') as file:
    for word, freq in sorted_words:
        file.write(f"{word}\t{freq}\n")

# Step 2: Load the dictionary and clean it up
df = pd.read_csv("dict.tsv", header=None, names=['gronings', 'dutch'], sep='\t')
df = df.drop_duplicates()

# Filter out rows where the Gronings and Dutch words are the same
df = df[df['gronings'] != df['dutch']]

# Filter out rows containing commas or spaces in either column
df_filtered = df[~df['gronings'].str.contains('[ ,]', na=False) & ~df['dutch'].str.contains('[ ,]', na=False)]

# Step 3: For each Dutch word with multiple Gronings translations, select the most frequent one or the shortest option
def select_best_gronings(dutch_word, gronings_options, sorted_words):
    max_freq = -1
    best_gronings = None
    for gronings_word in gronings_options:
        # Find the frequency of the Gronings word in the sorted_words list
        for word, freq in sorted_words:
            if word == gronings_word:
                if freq > max_freq:
                    max_freq = freq
                    best_gronings = gronings_word
                break
    # If no frequency match found, select the shortest option
    if best_gronings is None:
        best_gronings = min(gronings_options, key=len)
    return best_gronings

# Group by Dutch word and process each group
grouped = df_filtered.groupby('dutch')
filtered_rows = []
for dutch_word, group in grouped:
    if len(group) > 1:
        gronings_options = group['gronings'].tolist()
        best_gronings = select_best_gronings(dutch_word, gronings_options, sorted_words)
        filtered_rows.append({'gronings': best_gronings, 'dutch': dutch_word})
    else:
        filtered_rows.append(group.iloc[0].to_dict())

# Create a new DataFrame with the filtered rows
filtered_df = pd.DataFrame(filtered_rows)

# Save the filtered DataFrame to a new TSV file
filtered_df.to_csv("dict_clean.tsv", sep='\t', index=False, header=False)