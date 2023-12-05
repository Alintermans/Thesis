import os
import sys
from SongUtils import divide_song_into_paragraphs, get_syllable_count_of_sentence
from GPTLineGenerator2 import generate_parodie

initial_text_prompt = ""

#context = "I'm a lyric parodie writer this is an example of one and it's about that pineaple shouldn't be on pizza:\n"
context = "The new parodie will be about that pineaple shouldn't be on pizza\n"

original_song = ""

original_song_file_path = 'Experiments/GPT-2Constraints/songs/it_is_over_now-taylor_swift_small.txt'
original_song_file = open(original_song_file_path, 'r')
original_song += original_song_file.read()
original_song_file.close()

paragraphs = divide_song_into_paragraphs(original_song)

prompt = initial_text_prompt + context 


# generate line per line
generated_text = generate_parodie(prompt, paragraphs)
generated_lines = generated_text.split('\n')
current_line = 0
result = ''
total_lines = 0
correct_lines = 0

for (title, paragraph) in paragraphs:
    result += '\n' + title + '\n'
    for line in paragraph:
        result += generated_lines[current_line] + '\n'
        amount_of_syllables_original = get_syllable_count_of_sentence(line)
        amount_of_syllables_generated = get_syllable_count_of_sentence(generated_lines[current_line])
        if (amount_of_syllables_original == amount_of_syllables_generated):
            correct_lines += 1
        current_line += 1
        total_lines += 1

print("Paordie: \n", result)
print("Correct lines: ", correct_lines, " out of ", total_lines)

