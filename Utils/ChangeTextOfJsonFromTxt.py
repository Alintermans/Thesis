import json
import os

def changeTextOfJsonFromTxt(jsonPath, txtPath):
    with open(jsonPath, 'r') as f:
        data = json.load(f)
    with open(txtPath, 'r') as f:
        txt = f.read()
    data['parodie'] = txt

    with open(jsonPath, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    folder_path = 'Experiments/ConstrainedParodieGenerator/CallibrationExperiments/NoConstraints_edited'

    for folder in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder)):
            json_folder = os.path.join(folder_path, folder, 'None', 'json')
            txt_folder = os.path.join(folder_path, folder, 'None', 'text')

            for json_file in os.listdir(json_folder):
                if json_file.endswith('.json'):
                    json_path = os.path.join(json_folder, json_file)
                    txt_path = os.path.join(txt_folder, json_file.replace('.json', '.txt'))
                    txt_file = json_file.replace('.json', '.txt')
                    changeTextOfJsonFromTxt(json_path, txt_path)
                    print(f'Changed {json_file} from {txt_file}')