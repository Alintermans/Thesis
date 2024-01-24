# Author: Anton Lintermans | KU Leuven
# Written with the help of Github Copilot

import requests
import json
from bs4 import BeautifulSoup

JSON_FOLDER_PATH = "Songs/json"
TEXT_FOLDER_PATH = "Songs/text"

def extract_lyrics(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        print(f"Scraping {url}...")
        # Fetch artist, title, and original link from the HTML page
        artist = soup.find('a', class_=lambda value: value and 'Artist' in value).text.strip()
        title = soup.find('h1', class_=lambda value: value and 'Title' in value).text.strip()
        original_link = url
        
        # Find all div elements that have a class containing the word 'Lyrics'
        lyrics_divs = soup.find_all('div', class_=lambda value: value and 'Lyrics__Container' in value)

        lyrics = ""
        for div in lyrics_divs:
            for element in div:
                if element.name == 'br':  # If the element is a line break
                    lyrics += '\n'
                else:
                    lyrics += element.get_text('\n') 
        
        return lyrics.strip() if lyrics else "Lyrics not found.", artist, title, original_link

    except Exception as e:
        return f"An error occurred: {e}", None, None, None

def export_lyrics(url):
    lyrics, artist, title, original_link = extract_lyrics(url)

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

url_list = ["https://genius.com/Queen-bohemian-rhapsody-lyrics", "https://genius.com/Imagine-dragons-believer-lyrics",
"https://genius.com/Lady-gaga-and-bradley-cooper-shallow-lyrics", "https://genius.com/Arctic-monkeys-do-i-wanna-know-lyrics",
"https://genius.com/Leonard-cohen-hallelujah-lyrics", "https://genius.com/Hozier-take-me-to-church-lyrics",
"https://genius.com/Harry-styles-sign-of-the-times-lyrics","https://genius.com/The-chainsmokers-and-coldplay-something-just-like-this-lyrics", 
"https://genius.com/Harry-styles-as-it-was-lyrics", "https://genius.com/Billie-eilish-happier-than-ever-lyrics",
"https://genius.com/Taylor-swift-all-too-well-10-minute-version-taylors-version-live-acoustic-lyrics", "https://genius.com/The-killers-mr-brightside-lyrics",
"https://genius.com/Xxxtentacion-revenge-lyrics", "https://genius.com/Coldplay-viva-la-vida-lyrics",
"https://genius.com/Blink-182-i-miss-you-lyrics", "https://genius.com/Eminem-river-lyrics",
"https://genius.com/4-non-blondes-whats-up-lyrics", "https://genius.com/Rage-against-the-machine-killing-in-the-name-lyrics",
"https://genius.com/Journey-dont-stop-believin-lyrics", "https://genius.com/Nirvana-smells-like-teen-spirit-lyrics"
]

for url in url_list:
    export_lyrics(url)


