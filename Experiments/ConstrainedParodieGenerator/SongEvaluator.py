import json
import datetime
import os
from SongUtils import divide_song_into_paragraphs, get_pos_tags_of_line, similarity_of_pos_tags_sequences, get_syllable_count_of_sentence, _get_rhyming_lines,load_rhyming_dicts
from evaluate import load
import statistics


################# Init #################
perplexity = load("perplexity", module_type="metric", num_processes=4)
load_rhyming_dicts()
TOKENIZERS_PARALLELISM = True

################# Evaluation functions #################

def get_nb_lines_in_paragraphs(paragraphs):
    nb_lines = 0
    for paragraph in paragraphs:
        nb_lines += len(paragraph[1])
    return nb_lines

#The following function will check if the original and parodie song have the same number of paragraphs and lines and will return the paragraphs and lines that both the original and the parodie song have. 
def count_same_nb_lines_and_return_same_paragraphs(original_song_paragraps, parody_song_in_paragraphs):
    min_paragraph_len = min(len(original_song_paragraps), len(parody_song_in_paragraphs))
    original_song_nb_paragraphs = len(original_song_paragraps)
    parody_song_nb_paragraphs = len(parody_song_in_paragraphs)
    original_song_nb_lines = get_nb_lines_in_paragraphs(original_song_paragraps)
    parody_song_nb_lines = get_nb_lines_in_paragraphs(parody_song_in_paragraphs)
    correct_nb_paragraphs = 0
    correct_nb_lines = 0
    new_original_song_paragraphs = []
    new_parody_song_paragraphs = []
    for i in range(min_paragraph_len):
        min_len_lines = min(len(original_song_paragraps[i][1]), len(parody_song_in_paragraphs[i][1]))
        new_original_song_paragraphs.append(original_song_paragraps[i][1][:min_len_lines])
        new_parody_song_paragraphs.append(parody_song_in_paragraphs[i][1][:min_len_lines])
        correct_nb_lines += min_len_lines
        if len(original_song_paragraps[i][1]) == len(parody_song_in_paragraphs[i][1]):
            correct_nb_paragraphs += 1


    return original_song_nb_paragraphs, parody_song_nb_paragraphs, original_song_nb_lines, parody_song_nb_lines, new_original_song_paragraphs, new_parody_song_paragraphs, correct_nb_paragraphs, correct_nb_lines
        
    

def count_syllable_difference_per_line(original_song_paragraph, parody_song_paragraph):
    syllable_count_differences = []
    for i in range(len(original_song_paragraph)):
        for j in range(len(original_song_paragraph[i])):
            original_line = original_song_paragraph[i][j]
            parody_line = parody_song_paragraph[i][j]
            original_syllable_count = get_syllable_count_of_sentence(original_line)
            parody_syllable_count = get_syllable_count_of_sentence(parody_line)
            syllable_count_differences.append(abs(original_syllable_count - parody_syllable_count))
    return syllable_count_differences

def count_nb_line_pairs_match_rhyme_scheme(original_song_paragraph, parody_song_paragraph, rhyme_type):
    ##Assumes that the given paragraphs have the same number of lines
    matching_rhyme_pairs = 0
    expected_rhyming_pairs = 0
    rhyme_words_lengths = []
    for i in range(len(original_song_paragraph)):
        original_rhyming_lines = _get_rhyming_lines(original_song_paragraph[i], rhyme_type)
        parody_rhyming_lines = _get_rhyming_lines(parody_song_paragraph[i], rhyme_type)
        for j in range(len(original_rhyming_lines)):
            if original_rhyming_lines[j] is not None:
                expected_rhyming_pairs += 1
                if original_rhyming_lines[j] == parody_rhyming_lines[j]:
                    rhyme_words_lengths.append(get_syllable_count_of_sentence(parody_song_paragraph[i][j].split(' ')[-1]))
                    matching_rhyme_pairs += 1
    if len(rhyme_words_lengths) == 0:
        return matching_rhyme_pairs, expected_rhyming_pairs, 0
    return matching_rhyme_pairs, expected_rhyming_pairs, sum(rhyme_words_lengths)/len(rhyme_words_lengths)
    


def calculate_pos_tag_similarity(original_song_paragraph, parody_song_paragraph):
    pos_tag_similarities = []
    for i in range(len(original_song_paragraph)):
        for j in range(len(original_song_paragraph[i])):
            original_line = original_song_paragraph[i][j]
            parody_line = parody_song_paragraph[i][j]
            original_pos_tags = get_pos_tags_of_line(original_line)
            parody_pos_tags = get_pos_tags_of_line(parody_line)
            similarity = similarity_of_pos_tags_sequences(original_pos_tags, parody_pos_tags)
            pos_tag_similarities.append(similarity)
    return pos_tag_similarities


def calculate_perplexity(original_song_paragraph, parody_song_paragraph):
    #for each song the perplexity is calculated per line and then the mean over the whole song is calculated
    full_original_song_per_line = []
    full_parody_song_per_line = []
    for i in range(len(original_song_paragraph)):
        for j in range(len(original_song_paragraph[i])):
            original_line = original_song_paragraph[i][j]
            parody_line = parody_song_paragraph[i][j]
            full_original_song_per_line.append(original_line)   
            full_parody_song_per_line.append(parody_line)
    
    
    original_song_perplexity = perplexity.compute(predictions=full_original_song_per_line, model_id='gpt2')
    parody_song_perplexity = perplexity.compute(predictions=full_parody_song_per_line, model_id='gpt2')

    median_original_song_perplexity = statistics.median(original_song_perplexity['perplexities'])
    median_parody_song_perplexity = statistics.median(parody_song_perplexity['perplexities'])

    return median_original_song_perplexity, median_parody_song_perplexity

def calculate_overlap(original_song_paragraph, parody_song_paragraph):
    #the overlap between the original song and the parodie song is calculated by counting per line how many words are similar
    nb_overlap_words = 0
    nb_total_words = 0
    for i in range(len(original_song_paragraph)):
        for j in range(len(original_song_paragraph[i])):
            original_line = original_song_paragraph[i][j]
            parody_line = parody_song_paragraph[i][j]
            original_words = original_line.split(' ')
            parody_words = parody_line.split(' ')
            overlap = len([word for word in original_words if word in parody_words])
            nb_overlap_words += overlap
            nb_total_words += len(parody_words)
    
    return nb_overlap_words/nb_total_words


def calculate_repetition_score(song):
    #The repetition score is calculated by counting the number of unique words used in the song
    unique_words = set()
    nb_total_words = 0
    for i in range(len(song)):
        for j in range(len(song[i])):
            line = song[i][j]
            words = line.split(' ')
            unique_words.update(words)
            nb_total_words += len(words)
    return len(unique_words)/nb_total_words


def evaluate( parody_file_path, rhyme_type='perfect'):
    parody_song_file = None

    # Read the parodie song file
    with open(parody_file_path, "r") as f:
        parody_song_file = json.load(f)

    original_song_name = parody_song_file['original_song_title'].replace(' ', '_')
    original_song_artist = parody_song_file['original_song_artist'].replace(' ', '_')
    original_song_file_path = 'Songs/json/' + original_song_artist + '-' + original_song_name + '.json'
    # Read the original song file
    with open(original_song_file_path, "r") as f:
        original_song_file = json.load(f)
    

    original_song = original_song_file['lyrics']
    parody_song = parody_song_file['parodie']

    # Split in paragraphs
    original_song_in_paragraphs = divide_song_into_paragraphs(original_song)
    parody_song_in_paragraphs = divide_song_into_paragraphs(parody_song)

    # Count the number of paragraphs and lines in each song and return the paragraphs and lines that both the original and the parodie song both have 
    result = count_same_nb_lines_and_return_same_paragraphs(original_song_in_paragraphs, parody_song_in_paragraphs)
    original_song_nb_paragraphs = result[0]
    parody_song_nb_paragraphs = result[1]
    original_song_nb_lines = result[2]
    parody_song_nb_lines = result[3]
    original_song_in_paragraphs = result[4]
    parody_song_in_paragraphs = result[5]
    correct_nb_paragraphs = result[6]
    correct_nb_lines = result[7]


    print("Number of paragraphs in original song:", original_song_nb_paragraphs, "Number of paragraphs in parodie song:", parody_song_nb_paragraphs)
    print("Number of lines in original song:", original_song_nb_lines, "Number of lines in parodie song:", parody_song_nb_lines)

    # check the difference in syllables
    syllable_count_differences = count_syllable_difference_per_line(original_song_in_paragraphs, parody_song_in_paragraphs)
    nb_lines_correct_syllable_count = len([count for count in syllable_count_differences if count == 0])
    avg_syllable_count_difference = sum(syllable_count_differences) / len(syllable_count_differences)
    mean_deviation_syllable_count = sum([abs(count - avg_syllable_count_difference) for count in syllable_count_differences]) / len(syllable_count_differences)

    # check the rhyming 
    nb_matching_rhyme_pairs, nb_expected_rhyming_pairs, avg_rhyme_word_length = count_nb_line_pairs_match_rhyme_scheme(original_song_in_paragraphs, parody_song_in_paragraphs, rhyme_type)

    # check the pos tags similarity
    pos_tag_similarities = calculate_pos_tag_similarity(original_song_in_paragraphs, parody_song_in_paragraphs)
    nb_correct_pos_similarities = len([similarity for similarity in pos_tag_similarities if similarity == 1])
    avg_pos_tag_similarity = sum(pos_tag_similarities) / len(pos_tag_similarities)
    mean_deviation_pos_tag_similarity = sum([abs(similarity - avg_pos_tag_similarity) for similarity in pos_tag_similarities]) / len(pos_tag_similarities)

    # check the perplexity
    original_song_perplexity, parody_song_perplexity = calculate_perplexity(original_song_in_paragraphs, parody_song_in_paragraphs)

    # check the overlap with the original song
    overlap = calculate_overlap(original_song_in_paragraphs, parody_song_in_paragraphs)

    # check the repetition score, how many unique words are used
    original_song_repetition_score = calculate_repetition_score(original_song_in_paragraphs)
    parody_song_repetition_score = calculate_repetition_score(parody_song_in_paragraphs)

    # store the results in a json file
    results = {
        "original_song_name": original_song_file['title'],
        "parody_song_name": parody_song_file['original_song_title'],
        "original_song_file_path": original_song_file_path,
        "parody_song_file_path": parody_file_path,
        "original_song_nb_paragraphs": original_song_nb_paragraphs,
        "parody_song_nb_paragraphs": parody_song_nb_paragraphs,
        "correct_nb_paragraphs": correct_nb_paragraphs,
        "correct_nb_lines": correct_nb_lines,
        "original_song_nb_lines": original_song_nb_lines,
        "parody_song_nb_lines": parody_song_nb_lines,
        "nb_lines_correct_syllable_count": nb_lines_correct_syllable_count,
        "avg_syllable_count_difference": avg_syllable_count_difference,
        "mean_deviation_syllable_count": mean_deviation_syllable_count,
        "nb_matching_rhyme_pairs": nb_matching_rhyme_pairs,
        "nb_expected_rhyming_pairs": nb_expected_rhyming_pairs,
        "avg_rhyme_word_length": avg_rhyme_word_length,
        "nb_correct_pos_similarities": nb_correct_pos_similarities,
        "avg_pos_tag_similarity": avg_pos_tag_similarity,
        "mean_deviation_pos_tag_similarity": mean_deviation_pos_tag_similarity,
        "original_song_perplexity": original_song_perplexity,
        "parody_song_perplexity": parody_song_perplexity,
        "overlap": overlap,
        "original_song_repetition_score": original_song_repetition_score,
        "parody_song_repetition_score": parody_song_repetition_score

    }
    current_date = datetime.datetime.now()
    current_time = current_date.strftime("%Y-%m-%d_%H-%M-%S")
    language_model_name = parody_song_file['language_model_name']
    constrained_used = parody_song_file['constraints_used']
    directory = "Experiments/ConstrainedParodieGenerator/EvaluationResults"
    directory = directory + "/" + language_model_name.replace(' ', '_') + "/" + constrained_used.replace(' ', '_')
    file_name = parody_song_file['original_song_title'] + "_evaluation_results_" + current_time + ".json"
    file_path = directory + "/" + file_name

    #check if dir exists
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

    



if __name__ == "__main__":
    # # Ask the user for the original song file path (We expect the song to be in json format)
    # song_file_path = input("Enter the path to the original song file in json format: ")
    # # Ask the user for the path to the parody song file (We expect the song to be in json format)
    # parody_file_path = input("Enter the path to the parodie song file in json format: ")

    #song_file_path = 'Songs/json/Coldplay-Viva_La_Vida.json'
    parody_file_path = 'Experiments/ConstrainedParodieGenerator/CallibrationExperiments/PosConstraint/0/22/Llama 2 7B Chat/Syllable_Constraint_|_POS_Constraint_|_/json/Viva_La_Vida_parodie_30-04-2024_10h-02m-26s.json'
    
    
    evaluate(parody_file_path)





    


