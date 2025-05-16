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