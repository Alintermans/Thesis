import os
import sys
from SongUtils import divide_song_into_paragraphs
from GPTLineGenerator import generate_parodie_line

initial_text_prompt = "You're a parodie genrator that will write beatifull parodies and make sure that the syllable count of the parodie is the same as the original song\n"

context = "The new parodie will be about that pineaple shouldn't be on pizza\n"

original_song = ""

original_song_file_path = 'Experiments/GPT-2Constraints/songs/it_is_over_now-taylor_swift_small.txt'
original_song_file = open(original_song_file_path, 'r')
original_song += original_song_file.read()
original_song_file.close()

paragraps = divide_song_into_paragraphs(original_song)

original_song = "ORIGINAL SONG: \n\n" + original_song 

parodie = "\n\nAlready generated PARODIE: \n\n"
next_line_text = "The original line is: "
next_line_text_parodie = "The parodie line is: "

prompt = initial_text_prompt + context 

# generate line per line
for paragraph in paragraps:
    parodie += paragraph[0] + "\n"
    prompt += paragraph[0] + "\n"
    for line in paragraph[1]:
        new_prompt = prompt + parodie + next_line_text + line + "\n" + next_line_text_parodie
        result = generate_parodie_line(new_prompt, line) + "\n"
        parodie += result
        
        print(line, " | ",result)
    parodie += "\n"
    prompt += "\n"
    

print("Parodie: ", parodie)

