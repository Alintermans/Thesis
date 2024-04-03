songs = os.listdir(song_directory)
    # for song in songs:
    #     song_file_path = song_directory + song



########## Ranges For Hyperparameters To Test ##########
## General
num_possible_beams = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
do_samples = [True, False]

## Syllable Constraint
good_beamscore_multipliers_syllable = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
bad_beamscore_multipliers_syllable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

## Rhyming Constraint
rhyme_types = ['assonant', 'perfect', 'near']
top_k_rhyme_words = [10, 50, 100, 200, 500]
good_beamscore_multipliers_rhyme = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
good_beamscore_multipliers_assonant = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
continue_good_rhyme_multipliers = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
good_rhyming_token_multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
max_possible_syllable_counts = [1,2,3,4]

## POS Constraint
top_k_words_to_consider_for_pos = [100,200, 500,1000,2000,5000]
good_beamscore_multipliers_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
pos_similarity_limit_to_boosts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
good_token_multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
margin_of_similarity_with_new_tokens = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]
limilt_of_pos_similarity_to_satisfy_constraint = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.96, 0.97, 0.98, 0.99]



##Test syllable constraints
    # rhyming_constraint.disable()
    # pos_constraint.disable()
    # for lm_name in AVAILABLE_LMS.keys():
    #     if lm_name == 'GPT2':
    #         continue
    #     set_language_model(lm_name)
    #     for song in songs:
    #         song_file_path = song_directory + song
    #         for num_beam in num_possible_beams:
    #             set_num_beams(num_beam)
    #             for do_sample in do_samples:
    #                 for good_beamscore_multiplier_syllable in good_beamscore_multipliers_syllable:
    #                     for bad_beamscore_multiplier_syllable in bad_beamscore_multipliers_syllable:
    #                         syllable_constraint.set_hyperparameters(good_beamscore_multiplier=good_beamscore_multiplier_syllable, bad_beamscore_multiplier=bad_beamscore_multiplier_syllable)
    #                         chosen_hyper_parameters['SyllableConstraintLBL']['good_beamscore_multiplier'] = good_beamscore_multiplier_syllable
    #                         chosen_hyper_parameters['SyllableConstraintLBL']['bad_beamscore_multiplier'] = bad_beamscore_multiplier_syllable
    #                         constrained_used = "SyllableTest"
    #                         original_song, parody = generate_parodie(song_file_path, system_prompt, context, do_sample=do_sample, top_k=100, top_p=0.95, temperature=0.7, chosen_hyper_parameters=chosen_hyper_parameters, num_beams=num_beam, seed=42, constrained_used=constrained_used)


    
    #rhyming_constraint.disable()
    #pos_constraint.disable()

