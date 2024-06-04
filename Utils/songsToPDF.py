import os
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def generate_pdf_from_json_folder(folder_path, export_folder_path):
    pdf_path = os.path.join(export_folder_path, 'songs.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

        title = data['original_song_title']
        artist = data['original_song_artist']

        original_song_name = data['original_song_title'].replace(' ', '_')
        original_song_artist = data['original_song_artist'].replace(' ', '_')
        original_song_file_path = 'Songs/json/' + original_song_artist + '-' + original_song_name + '.json'
        # Read the original song file
        with open(original_song_file_path, "r") as f:
            original_song_file = json.load(f)
        

        original_song = original_song_file['lyrics']
        parody_song = data['parodie']
        

        # Draw title and artist
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 750, f"Title: {title}")
        c.drawString(50, 720, f"Artist: {artist}")

        # Draw original song
        c.setFont("Helvetica", 5)
        c.drawString(50, 680, "Original Song:")
        start = 660
        for line in original_song.split('\n'):
            c.drawString(50, start, f"{line}")
            start -= 6
        

        # Draw parody song
        c.drawString(300, 680, "Parody Song:")
        start = 660
        for line in parody_song.split('\n'):
            c.drawString(300, start, f"{line}")
            start -= 6

        # Add page break
        c.showPage()

    c.save()
    print(f"PDF generated successfully at {pdf_path}")

# Usage example
export_folder_path = 'exportPDF/'
if not os.path.exists(export_folder_path):
    os.makedirs(export_folder_path)

folder_path = 'Experiments/ConstrainedParodieGenerator/CallibrationExperiments/AllChat/1/Llama 2 7B Chat/Syllable_Constraint_|_Rhyming_Constraint_|_POS_Constraint_|_/json'

generate_pdf_from_json_folder(folder_path, export_folder_path)