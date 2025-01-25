# Just wanna see if the words from readability's word lists are all in tatoeba

import readability
basicwords_nl = readability.langdata.basicwords_nl #it's a set so order is lost. see the order on their github

cleaned_basicwords_nl = [word for word in basicwords_nl if not any(char.isdigit() for char in word)]

print(cleaned_basicwords_nl)

# ... I haven't finished this lol