
import os
import glob
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

test_list_file = open('test_list.csv', 'w')
test_list_file.write('image_name, class\n')
test_path = os.path.join('..', 'input', 'test', '*.jpg')
files = glob.glob(test_path)
files = sorted(files, key=natural_keys)
for file in files:
    file_name = str(file)
    test_list_file.write(file_name[14:])
    test_list_file.write('\n')

