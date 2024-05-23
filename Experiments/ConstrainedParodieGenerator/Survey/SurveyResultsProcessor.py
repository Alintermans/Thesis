# import pandas as pd
# import matplotlib.pyplot as plt

# # Path to the CSV file
# csv_file = 'Experiments/ConstrainedParodieGenerator/Survey/results-survey655765.csv'

# # Columns to be read
# columns = ['G1Q1[SQ001]', 'G1Q1[SQ002]', 'G1Q1[SQ003]', 'G1Q2[SQ001]', 'G1Q2[SQ002]', 'G1Q2[SQ003]', 'G1Q3[SQ001]', 'G1Q3[SQ002]', 'G1Q3[SQ003]', 'G1Q4[SQ001]', 'G1Q4[SQ002]', 'G1Q4[SQ003]', 'G1Q5[SQ001]', 'G1Q5[SQ002]', 'G1Q5[SQ003]', 'G2Q1[SQ001]', 'G2Q1[SQ002]', 'G2Q1[SQ003]', 'G2Q2[SQ001]', 'G2Q2[SQ002]', 'G2Q2[SQ003]', 'G2Q3[SQ001]', 'G2Q3[SQ002]', 'G2Q3[SQ003]', 'G2Q4[SQ001]', 'G2Q4[SQ002]', 'G2Q4[SQ003]', 'G2Q5[SQ001]', 'G2Q5[SQ002]', 'G2Q5[SQ003]', 'G3Q1[SQ001]', 'G3Q1[SQ002]', 'G3Q1[SQ003]', 'G3Q2[SQ001]', 'G3Q2[SQ002]', 'G3Q2[SQ003]', 'G3Q3[SQ001]', 'G3Q3[SQ002]', 'G3Q3[SQ003]', 'G3Q4[SQ001]', 'G3Q4[SQ002]', 'G3Q4[SQ003]', 'G3Q5[SQ001]', 'G3Q5[SQ002]', 'G3Q5[SQ003]', 'G4Q1[SQ001]', 'G4Q1[SQ002]', 'G4Q1[SQ003]', 'G4Q2[SQ001]', 'G4Q2[SQ002]', 'G4Q2[SQ003]', 'G4Q3[SQ001]', 'G4Q3[SQ002]', 'G4Q3[SQ003]', 'G4Q4[SQ001]', 'G4Q4[SQ002]', 'G4Q4[SQ003]', 'G4Q5[SQ001]', 'G4Q5[SQ002]', 'G4Q5[SQ003]']

# # Read the CSV file and select the specified columns
# df = pd.read_csv(csv_file, usecols=columns)

# no_constraints = [0,0,0]
# with_constraints = [0,0,0]
# total = 0
# questions = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# questions_with_constraints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# questions_no_constraints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

# # for every general question so for G1Q1[SQ001] is this Q1 has three subquestions SQ001, SQ002, SQ003 
# # and for G1Q2[SQ001] is this Q2 has three subquestions SQ001, SQ002, SQ003
# # for each question on each row we are going to check if there is an answer for the subqesutions, if there is answer ('Paordy 1' or 'Parody 2') we will increase the correct variable
# # for unevenquestions 'Parody 1' belongs to no_constraints and 'Parody 2' belongs to with_constraints and for even questions 'Parody 1' belongs to with_constraints and 'Parody 2' belongs to no_constraints

# for index, row in df.iterrows():
#     for i in range(1, 5):
#         for j in range(1, 6):
#             for k in range(1, 4):
#                 question = 'G' + str(i) + 'Q' + str(j) + '[SQ00' + str(k) + ']'
#                 if row[question] == 'Parody 1':
#                     if j % 2 == 0:
#                         with_constraints[k-1] += 1
#                         questions_with_constraints[(i-1)*5 + j-1] += 1
#                     else:
#                         no_constraints[k-1] += 1
#                         questions_no_constraints[(i-1)*5 + j-1] += 1
#                     total += 1
#                     questions[(i-1)*5 + j-1] += 1
#                 elif row[question] == 'Parody 2':
#                     if j % 2 == 0:
#                         no_constraints[k-1] += 1
#                         questions_no_constraints[(i-1)*5 + j-1] += 1
#                     else:
#                         with_constraints[k-1] += 1
#                         questions_with_constraints[(i-1)*5 + j-1] += 1
#                     total += 1
#                     questions[(i-1)*5 + j-1] += 1

# if total % 3 != 0:
#     print('Error: Total number of answers is not a multiple of 3')

# total = total // 3
# questions = [x // 3 for x in questions]
# questions_with_constraints = [x // 3 for x in questions_with_constraints]
# questions_no_constraints = [x // 3 for x in questions_no_constraints]

# print('No constraints:')
# print('SQ001: ' + str(no_constraints[0] / total))
# print('SQ002: ' + str(no_constraints[1] / total))
# print('SQ003: ' + str(no_constraints[2] / total))

# print('With constraints:')
# print('SQ001: ' + str(with_constraints[0] / total))
# print('SQ002: ' + str(with_constraints[1] / total))
# print('SQ003: ' + str(with_constraints[2] / total))

# print('Total: ' + str(total))
# print('Questions: ' + str(questions))
# print('Questions with constraints: ' + str(questions_with_constraints))
# print('Questions no constraints: ' + str(questions_no_constraints))

# def plot_questions(questions):
#     labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20']
#     title = 'Survey questions'

#     questions = [x / total for x in questions]

#     x = range(len(labels))

#     fig, ax = plt.subplots()
#     ax.bar(x, questions, width=0.4)

#     ax.set_xticks([i for i in x])
#     ax.set_xticklabels(labels)
#     ax.set_title(title)
#     ax.set_ylabel('Percentage')

#     plt.savefig('Experiments/ConstrainedParodieGenerator/Survey/questions-survey655765.png', dpi=300)
#     plt.show()

# plot_questions(questions)



# # SQ001 is about Coherency, SQ002 is about Singability, SQ003 is about Humor
# # the following functions will make it into a nice graph

# def plot_results(no_constraints, with_constraints):
#     labels = ['Coherency', 'Singability', 'Humor']
#     title = 'Survey results'

#     no_constraints = [x / total for x in no_constraints]
#     with_constraints = [x / total for x in with_constraints]

#     x = range(len(labels))

#     fig, ax = plt.subplots()
#     ax.bar(x, no_constraints, width=0.4, label='No constraints')
#     ax.bar([i + 0.4 for i in x], with_constraints, width=0.4, label='With constraints')

#     ax.set_xticks([i + 0.2 for i in x])
#     ax.set_xticklabels(labels)
#     ax.set_title(title)
#     ax.set_ylabel('Percentage')

#     ax.legend()
#     plt.savefig('Experiments/ConstrainedParodieGenerator/Survey/results-survey655765.png', dpi=300)
#     plt.show()

# #The following funciton puts the results in a nice table for latex in a .tex file

# def results_to_latex_table(no_constraints, with_constraints):
#     no_constraints = [round(x / total, 2) for x in no_constraints]
#     with_constraints = [round(x / total, 2) for x in with_constraints]
#     title = 'Survey results'

#     with open('Experiments/ConstrainedParodieGenerator/Survey/results-survey655765.tex', 'w') as f:
#         f.write('\\begin{table}[]\n')
#         f.write('\\centering\n')
#         f.write('\\begin{tabular}{|c|c|c|}\n')
#         f.write('\\hline\n')
#         f.write(' & No constraints & With constraints \\\\\n')
#         f.write('\\hline\n')
#         f.write('Coherency & ' + str(no_constraints[0]) + ' & ' + str(with_constraints[0]) + ' \\\\\n')
#         f.write('Singability & ' + str(no_constraints[1]) + ' & ' + str(with_constraints[1]) + ' \\\\\n')
#         f.write('Humor & ' + str(no_constraints[2]) + ' & ' + str(with_constraints[2]) + ' \\\\\n')
#         f.write('\\hline\n')
#         f.write('\\end{tabular}\n')
#         f.write('\\caption{' + title + '}\n')
#         f.write('\\end{table}\n')







# plot_results(no_constraints, with_constraints)
# results_to_latex_table(no_constraints, with_constraints)
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Path to the CSV file
csv_file = 'Experiments/ConstrainedParodieGenerator/Survey/results-survey655765.csv'

# Columns to be read
columns = ['G1Q1[SQ001]', 'G1Q1[SQ002]', 'G1Q1[SQ003]', 'G1Q2[SQ001]', 'G1Q2[SQ002]', 'G1Q2[SQ003]', 
           'G1Q3[SQ001]', 'G1Q3[SQ002]', 'G1Q3[SQ003]', 'G1Q4[SQ001]', 'G1Q4[SQ002]', 'G1Q4[SQ003]', 
           'G1Q5[SQ001]', 'G1Q5[SQ002]', 'G1Q5[SQ003]', 'G2Q1[SQ001]', 'G2Q1[SQ002]', 'G2Q1[SQ003]', 
           'G2Q2[SQ001]', 'G2Q2[SQ002]', 'G2Q2[SQ003]', 'G2Q3[SQ001]', 'G2Q3[SQ002]', 'G2Q3[SQ003]', 
           'G2Q4[SQ001]', 'G2Q4[SQ002]', 'G2Q4[SQ003]', 'G2Q5[SQ001]', 'G2Q5[SQ002]', 'G2Q5[SQ003]', 
           'G3Q1[SQ001]', 'G3Q1[SQ002]', 'G3Q1[SQ003]', 'G3Q2[SQ001]', 'G3Q2[SQ002]', 'G3Q2[SQ003]', 
           'G3Q3[SQ001]', 'G3Q3[SQ002]', 'G3Q3[SQ003]', 'G3Q4[SQ001]', 'G3Q4[SQ002]', 'G3Q4[SQ003]', 
           'G3Q5[SQ001]', 'G3Q5[SQ002]', 'G3Q5[SQ003]', 'G4Q1[SQ001]', 'G4Q1[SQ002]', 'G4Q1[SQ003]', 
           'G4Q2[SQ001]', 'G4Q2[SQ002]', 'G4Q2[SQ003]', 'G4Q3[SQ001]', 'G4Q3[SQ002]', 'G4Q3[SQ003]', 
           'G4Q4[SQ001]', 'G4Q4[SQ002]', 'G4Q4[SQ003]', 'G4Q5[SQ001]', 'G4Q5[SQ002]', 'G4Q5[SQ003]']

# Read the CSV file and select the specified columns
df = pd.read_csv(csv_file, usecols=columns)

no_constraints = [0, 0, 0]
with_constraints = [0, 0, 0]
total = 0
questions = [0] * 20
questions_with_constraints = [0] * 20
questions_no_constraints = [0] * 20

for index, row in df.iterrows():
    for i in range(1, 5):
        for j in range(1, 6):
            for k in range(1, 4):
                question = 'G' + str(i) + 'Q' + str(j) + '[SQ00' + str(k) + ']'
                if row[question] == 'Parody 1':
                    if j % 2 == 0:
                        with_constraints[k - 1] += 1
                        questions_with_constraints[(i - 1) * 5 + j - 1] += 1
                    else:
                        no_constraints[k - 1] += 1
                        questions_no_constraints[(i - 1) * 5 + j - 1] += 1
                    total += 1
                    questions[(i - 1) * 5 + j - 1] += 1
                elif row[question] == 'Parody 2':
                    if j % 2 == 0:
                        no_constraints[k - 1] += 1
                        questions_no_constraints[(i - 1) * 5 + j - 1] += 1
                    else:
                        with_constraints[k - 1] += 1
                        questions_with_constraints[(i - 1) * 5 + j - 1] += 1
                    total += 1
                    questions[(i - 1) * 5 + j - 1] += 1

if total % 3 != 0:
    print('Error: Total number of answers is not a multiple of 3')

total = total // 3
questions = [x // 3 for x in questions]
questions_with_constraints = [x // 3 for x in questions_with_constraints]
questions_no_constraints = [x // 3 for x in questions_no_constraints]

print('No constraints:')
print('SQ001: ' + str(no_constraints[0] / total))
print('SQ002: ' + str(no_constraints[1] / total))
print('SQ003: ' + str(no_constraints[2] / total))

print('With constraints:')
print('SQ001: ' + str(with_constraints[0] / total))
print('SQ002: ' + str(with_constraints[1] / total))
print('SQ003: ' + str(with_constraints[2] / total))

print('Total: ' + str(total))
print('Questions: ' + str(questions))
print('Questions with constraints: ' + str(questions_with_constraints))
print('Questions no constraints: ' + str(questions_no_constraints))

# Function to perform chi-square test for each subquestion
def perform_chi_square(no_constraints, with_constraints):
    labels = ['Coherency', 'Singability', 'Humor']
    for i in range(3):
        contingency_table = [[no_constraints[i], with_constraints[i]], [total - no_constraints[i], total - with_constraints[i]]]
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        print(f'{labels[i]}: Chi2 = {chi2}, p-value = {p} ({p < 0.05})  (dof = {dof}) (expected = {ex})') 

perform_chi_square(no_constraints, with_constraints)

def plot_questions(questions):
    labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20']
    title = 'Survey questions'

    questions = [x / total for x in questions]

    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, questions, width=0.4)

    ax.set_xticks([i for i in x])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel('Percentage')

    plt.savefig('Experiments/ConstrainedParodieGenerator/Survey/questions-survey655765.png', dpi=300)
    plt.show()

plot_questions(questions)

# SQ001 is about Coherency, SQ002 is about Singability, SQ003 is about Humor
# the following functions will make it into a nice graph

def plot_results(no_constraints, with_constraints):
    labels = ['Coherency', 'Singability', 'Humor']
    title = 'Survey results'

    no_constraints = [x / total for x in no_constraints]
    with_constraints = [x / total for x in with_constraints]

    x = range(len(labels))

    fig, ax = plt.subplots()
    ax.bar(x, no_constraints, width=0.4, label='No constraints')
    ax.bar([i + 0.4 for i in x], with_constraints, width=0.4, label='With constraints')

    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel('Percentage')

    ax.legend()
    plt.savefig('Experiments/ConstrainedParodieGenerator/Survey/results-survey655765.png', dpi=300)
    plt.show()

# The following function puts the results in a nice table for latex in a .tex file
def results_to_latex_table(no_constraints, with_constraints):
    no_constraints = [round(x / total, 2) for x in no_constraints]
    with_constraints = [round(x / total, 2) for x in with_constraints]
    title = 'Survey results'

    with open('Experiments/ConstrainedParodieGenerator/Survey/results-survey655765.tex', 'w') as f:
        f.write('\\begin{table}[]\n')
        f.write('\\centering\n')
        f.write('\\begin{tabular}{|c|c|c|}\n')
        f.write('\\hline\n')
        f.write(' & No constraints & With constraints \\\\\n')
        f.write('\\hline\n')
        f.write('Coherency & ' + str(no_constraints[0]) + ' & ' + str(with_constraints[0]) + ' \\\\\n')
        f.write('Singability & ' + str(no_constraints[1]) + ' & ' + str(with_constraints[1]) + ' \\\\\n')
        f.write('Humor & ' + str(no_constraints[2]) + ' & ' + str(with_constraints[2]) + ' \\\\\n')
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\caption{' + title + '}\n')
        f.write('\\end{table}\n')

plot_results(no_constraints, with_constraints)
results_to_latex_table(no_constraints, with_constraints)
