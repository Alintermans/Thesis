
def divide_song_into_paragraphs(song):
    paragraphs = []
    current_paragraph = []
    current_paragraph_name = ""
    first_done = False
    for line in song.split("\n"):
        if line.startswith("["):
            if first_done:
                paragraphs.append((current_paragraph_name, current_paragraph))
                current_paragraph = []
            else:
                first_done = True
            current_paragraph_name = line
        elif line == '':
                continue
        else:
            current_paragraph.append(line)
    paragraphs.append((current_paragraph_name, current_paragraph))
    return paragraphs

