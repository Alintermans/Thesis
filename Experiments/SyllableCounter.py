import nltk
from nltk.corpus import cmudict
from multiprocessing import Pool
import multiprocessing

d = cmudict.dict()

# Define function to count the number of syllables in a word
def count_syllables(word):
    #if the string is a puntcuation mark, return 0
    if word in [".", ",", "!", "?", ";", ":", "-", "'", "\"", "(", ")", "[", "]", "{", "}"]:
        return 0

    
    try:
        result =  [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        # if word not found in cmudict
        result =  count_syllables_hard(word)
    return result
def count_syllables_hard(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
        


# Define function to count the number of syllables in a line of text
def count_syllables_line(line):
    words = nltk.word_tokenize(line)
    result = sum(count_syllables(word) for word in words)
    #print("Line:", line, "Syllables:", result)
    return result

# Define a function to count matching lines in a paragraph
def count_matching_lines(paragraph):
    lines1 = paragraph[0]
    lines2 = paragraph[1]

    # Remove empty lines from the lists of lines
    lines1 = [line for line in lines1 if line != ""]
    lines2 = [line for line in lines2 if line != ""]

    # Print the number of lines in each paragraph
    # print("Number of lines in paragraph of original song:", len(lines1))
    # print("Number of lines in paragraph of parodie song:", len(lines2))

    syllable_counts1 = list(map(count_syllables_line, lines1))
    syllable_counts2 = list(map(count_syllables_line, lines2))
    # Count the number of matching lines in the paragraph
    matching_lines = sum(
        1 for syllables1, syllables2 in zip(syllable_counts1, syllable_counts2) if syllables1 == syllables2
    )

    syllable_count_differences = []
    
    for syllables1, syllables2 in zip(syllable_counts1, syllable_counts2):
        syllable_count_differences.append(abs(syllables1 - syllables2))
        
    
    return matching_lines, syllable_count_differences



if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('cmudict')
    #Ask the user for the path to the input file, which contains two songs one ORIGINAL and one PARODIE, the song has a structure dictated by [] tags and the lines are separated by \n characters
    #The program will count the number of lines with matching syllable counts per paragraph and print the result
    path = input("Enter the path to the input file: ")

    # Read the input file
    with open(path, "r") as f:
        text = f.read()

    # Split the text into to the original and parodie song by the ORIGINAL: and PARODIE: tags
    text = text.split("ORIGINAL:")[1]
    text = text.split("PARODIE:")
    lines1 = text[0].split("\n")
    lines2 = text[1].split("\n")

    # Remove empty lines from the lists of lines
    lines1 = [line for line in lines1 if line != ""]
    lines2 = [line for line in lines2 if line != ""]

    # dived the lines into paragraphs by the [] tags the text within the tags is ignored
    paragraphs1 =  []
    paragraphs2 =  []

    for line in lines1:
        if line.startswith("["):
            paragraphs1.append([])
        else:
            paragraphs1[-1].append(line)
    for line in lines2:
        if line.startswith("["):
            paragraphs2.append([])
        else:
            paragraphs2[-1].append(line)

    # Remove empty paragraphs from the lists of paragraphs
    paragraphs1 = [paragraph for paragraph in paragraphs1 if paragraph != []]
    paragraphs2 = [paragraph for paragraph in paragraphs2 if paragraph != []]

    # Print the number of paragraphs in each song
    print("Number of paragraphs in original song:", len(paragraphs1))
    print("Number of paragraphs in parodie song:", len(paragraphs2))

    # Print the number of lines in each song
    print("Number of lines in original song:", len(lines1))
    print("Number of lines in parodie song:", len(lines2))



    #compare for each paragraph the number of lines with matching syllable counts and print the total number of matching lines compared to the total number of lines
    matching_lines = 0
    total_lines = 0



    # Use multiprocessing to count matching lines in each paragraph
    pool = Pool(multiprocessing.cpu_count())

    matching_lines_list = pool.map(count_matching_lines, zip(paragraphs1, paragraphs2))
    matching_lines_list, syllable_count_differences_list = zip(*matching_lines_list)
    print("Matching lines list:", matching_lines_list)
    print("Syllable count differences list:", syllable_count_differences_list)
    # make statistics about the syllable count differences
    syllable_count_differences = []
    for paragraph in syllable_count_differences_list:
        syllable_count_differences += paragraph

    print("Average syllable count difference:", sum(syllable_count_differences)/len(syllable_count_differences))
    print("Max syllable count difference:", max(syllable_count_differences))
    print("Min syllable count difference:", min(syllable_count_differences))
    print("Median syllable count difference:", sorted(syllable_count_differences)[len(syllable_count_differences)//2])




    # Print the total number of lines with matching syllable counts
    total_matching_lines = sum(matching_lines_list)
    total_lines = sum(min(len(paragraph1), len(paragraph2)) for paragraph1, paragraph2 in zip(paragraphs1, paragraphs2))
    total_matching_paragraphs = sum(1 for paragraph1, paragraph2 in zip(paragraphs1, paragraphs2) if len(paragraph1) == len(paragraph2))
    print("Total matching lines:", total_matching_lines)
    print("Total lines:", total_lines)
    print("Total matching paragraphs:", total_matching_paragraphs)
    print("Total paragraphs:", len(paragraphs1))

