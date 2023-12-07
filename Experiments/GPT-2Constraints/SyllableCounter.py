import nltk
from nltk.corpus import cmudict
from multiprocessing import Pool
import multiprocessing
from SongUtils import count_matching_lines_on_syllable_amount

if __name__ == '__main__':
    #Ask the user for the path to the input file, which contains two songs one ORIGINAL and one PARODIE, the song has a structure dictated by [] tags and the lines are separated by \n characters
    #The program will count the number of lines with matching syllable counts per paragraph and print the result
    path = input("Enter the path to the input file: ")

    # Read the input file
    with open(path, "r") as f:
        text = f.read()
    # Split the text into to the original and parodie song by the ORIGINAL: and PARODIE: tags
    text = text.split("ORIGINAL:")[1]
    text = text.split("PARODIE:")
    original = text[0].split("\n")
    parodie = text[1].split("\n")

    lines1 = [line for line in original if line != ""]
    lines2 = [line for line in parodie if line != ""]

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

    matching_lines_list = pool.map(count_matching_lines_on_syllable_amount, zip(paragraphs1, paragraphs2))
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

