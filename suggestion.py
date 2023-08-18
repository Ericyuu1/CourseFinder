import difflib
import time

def checkspell(query):
    with open('words.txt', 'r') as file:
        words_string = file.read()

    words_list = [f'{word.strip()}' for word in words_string.split(',')]

    # List of possible correct spellings
    correct_spellings = words_list

    # Word to check
    word_to_check = query

    # Find the closest match
    closest_match = difflib.get_close_matches(word_to_check, correct_spellings, n=1)

    end=time.time()
    # Check if there is a close match
    if closest_match:
        return closest_match[0]
    else:
        return "no match found"
