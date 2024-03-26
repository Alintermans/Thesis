import os
import json
import datetime
from datetime import date, datetime


def convert_lyrics(filename_txt, title, original_song_artist, language_model, original_song_file_path):
    # Load lyrics from text file
    if not os.path.exists(filename_txt):
        print(f"File {filename_txt} not found.")
        return

    with open(filename_txt, 'r') as text_file:
        lyrics = text_file.read()
    

    if language_model and title and lyrics != "Lyrics not found.":
        print(f"Exporting {language_model} - {original_song_artist} - {title}...")
        language_model_stripped = language_model.replace(' ', '_')
        title_stripped = title.replace(' ', '_')
        artist_stripped = original_song_artist.replace(' ', '_')

        # Export lyrics to JSON file

        date_today = date.today().strftime("%d-%m-%Y")
        time = datetime.now().strftime("%Hh-%Mm-%Ss")

        json_data = {
        "original_song_title": title,
        "original_song_artist": original_song_artist,
        "language_model_name": language_model,
        "system_prompt": "",
        "context": "",
        "prompt": "",
        "constraints_used": "None",
        "chosen_hyper_parameters": "",
        "num_beams": "",
        "seed": "",
        "way_of_generation": "",
        "decoding_method": "",
        "state": "",
        "date": date_today,
        "time": time,
        "parodie": lyrics
    }
        json_filename = f"{language_model_stripped}-{artist_stripped}-{title_stripped}.json"
        with open(f"{json_filename}", 'w') as json_file:
            json.dump(json_data, json_file,indent=4)

        print(f"Exported {original_song_artist} - {title} to {json_filename}")


if __name__ == "__main__":
    print("Give me the path to the txt file")
    filename_txt = input()
    print("Give me the title")
    title = input()
    print("Give me the name of the original artist")
    original_song_artist = input()
    print("Give me the name of the language model used")
    language_model = input()
    print("Give me the path to the json of the original song, if none available press enter.")
    original_song_file_path = input()


    convert_lyrics(filename_txt, title, original_song_artist, language_model, original_song_file_path)

    print("Done!")

