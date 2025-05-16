import numpy as np
# Reading and processing text
with open('1268-0.txt', 'r', encoding="utf-8") as fp:
    text=fp.read()

start_indx = text.find('THE MYSTERIOUS ISLAND')
end_indx = text.find('End of the Project Gutenberg')
text = text[start_indx:end_indx]
char_set = set(text)
print('start index: ', start_indx)
print('end index: ', end_indx)
print('Total Length:', len(text))
print('Unique Characters:', len(char_set))

# We convert the character into integer and vice versa
# Building the dictionary
chars_sorted = sorted(char_set)
char2int = {ch:i for i,ch in enumerate(chars_sorted)}
char_array = np.array(chars_sorted)
text_encoded = np.array([char2int[ch] for ch in text], dtype=np.int32)
print('Text encoded shape:', text_encoded.shape)
print(text[:15], '== Encoding ==>', text_encoded.shape)
print(text_encoded[15:21], '== Reverse ==>', ''.join(char_array[text_encoded[15:21]]))