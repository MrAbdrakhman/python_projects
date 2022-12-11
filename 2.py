def word_search(doc_list, keyword):
    # list to hold the indices of matching documents
    indices = []
    # Iterate through the indices (i) and elements (doc) of documents
    for i, doc in enumerate(doc_list):
        # Split the string doc into a list of words (according to whitespace)
        print (i, doc)
        tokens = doc.split()
        print (tokens)
        # Make a transformed list where we 'normalize' each word to facilitate matching.
        # Periods and commas are removed from the end of each word, and it's set to all lowercase.
        normalized = [token.rstrip('.,').lower() for token in tokens]
        print (normalized)
        # Is there a match? If so, update the list of matching indices.
        for i, g in enumerate(keyword):
            token = doc.split()
            print(token)
            # Make a transformed list where we 'normalize' each word to facilitate matching.
            # Periods and commas are removed from the end of each word, and it's set to all lowercase.
            normalized1 = [tok.rstrip('.,').lower() for tok in token]
            if keyword.lower() in normalized1:
                indices.append(i)
    return indices

doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]
keyword = ['casino', 'they']

print(word_search(doc_list, keyword))