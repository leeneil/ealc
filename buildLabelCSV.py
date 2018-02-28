import os
import io
import glob
import argparse

DEFAULT_PATH1_DATA = os.path.join('data')
DEFAULT_PATH2_DATA = os.path.join('ted/samples_128')
DEFAULT_NUM_DATA = 500000
DICT_LANGUAGE_LABEL = {'tw':0, 'cn':1, 'jp':2, 'kr':3}


def label_a_language(labels_csv, language, path1_data, path2_data, num_data):
    files = glob.glob(os.path.join(path1_data, language,path2_data, '*.png'))
    filesList = [files[i] for i in range(num_data)]
    for file_path in filesList:
        labels_csv.write(u'{},{}\n'.format(file_path, DICT_LANGUAGE_LABEL[language]))
    print('----- Finish labeling: ' + language + ' -----')

def label_all_languages(path1_data, path2_data, num_data):
    labels_csv = io.open(os.path.join(path1_data, 'labels-map.csv'), 'w', encoding='utf-8')
    label_a_language(labels_csv, 'tw',path1_data, path2_data, num_data)
    label_a_language(labels_csv, 'cn',path1_data, path2_data, num_data)
    label_a_language(labels_csv, 'jp',path1_data, path2_data, num_data)
    label_a_language(labels_csv, 'kr',path1_data, path2_data, num_data)
    labels_csv.close()
    print('----- Finish labeling all languages -----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1-data', type=str, dest='path1_data',
                        default=DEFAULT_PATH1_DATA, help='Path of folder that contains data.')
    parser.add_argument('--path2-data', type=str, dest='path2_data',
                        default=DEFAULT_PATH2_DATA, help='Path of folder that contains data.')                        
    parser.add_argument('--num-data', type=str, dest='num_data',
                        default=DEFAULT_NUM_DATA, help='Number of data you want to include in training.')                    
    args = parser.parse_args()
    label_all_languages(args.path1_data, args.path2_data, int(args.num_data))
    
    
    
    
