import os
import sys
from SongUtils import divide_song_into_paragraphs
from GPTLineGenerator import generate_parodie_line

initial_text_prompt = "You're a parodie genrator that will write beatifull parodies and make sure that the syllable count of the parodie is the same as the original song\n"

context = "The new parodie will be about that pineaple shouldn't be on pizza\n"

original_song = ""

original_song_file_path = 'Experiments/GPT-2Constraints/songs/it_is_over_now-taylor_swift.txt'
original_song_file = open(original_song_file_path, 'r')
original_song += original_song_file.read()
original_song_file.close()

paragraps = divide_song_into_paragraphs(original_song)

original_song = "ORIGINAL SONG: \n\n" + original_song 

parodie = "\n\nPARODIE: \n\n"

prompt = initial_text_prompt + context + original_song + parodie

# generate line per line
for paragraph in paragraps:
    parodie += paragraph[0] + "\n"
    prompt += paragraph[0] + "\n"
    for line in paragraph[1]:
        result = generate_parodie_line(prompt, line) + "\n"
        parodie += result
        prompt += result
        print(line, " | ",result)
    parodie += "\n"
    prompt += "\n"

print("Paordie: ", parodie)

