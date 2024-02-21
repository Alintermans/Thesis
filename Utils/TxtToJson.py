import os
import json


JSON_FOLDER_PATH = "Songs/json"
TEXT_FOLDER_PATH = "Songs/text"

def convert_lyrics(filename_txt, artisy, title, original_link):
    # Load lyrics from text file
    if not os.path.exists(filename_txt):
        print(f"File {filename_txt} not found.")
        return

    with open(filename_txt, 'r') as text_file:
        lyrics = text_file.read()
    

    if artist and title and lyrics != "Lyrics not found.":
        print(f"Exporting {artist} - {title}...")
        artist_stripped = artist.replace(' ', '_')
        title_stripped = title.replace(' ', '_')
        # Export lyrics to text file
        text_filename = f"{artist_stripped}-{title_stripped}.txt"
        with open(f"{TEXT_FOLDER_PATH}/{text_filename}", 'w') as text_file:
            text_file.write(lyrics)

        # Export lyrics to JSON file
        json_data = {
            "artist": artist,
            "title": title,
            "original_link": original_link,
            "lyrics": lyrics
        }
        json_filename = f"{artist_stripped}-{title_stripped}.json"
        with open(f"{JSON_FOLDER_PATH}/{json_filename}", 'w') as json_file:
            json.dump(json_data, json_file)

        print(f"Exported {artist} - {title} to {text_filename} and {json_filename}")


if __name__ == "__main__":
    print("Give me the path to the txt file")
    filename_txt = input()
    print("Give me the artist")
    artist = input()
    print("Give me the title")
    title = input()
    print("Give me the original link, if none available press enter.")
    original_link = input()

    convert_lyrics(filename_txt, artist, title, original_link)

    print("Done!")

