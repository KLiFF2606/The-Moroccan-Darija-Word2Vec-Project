from bs4 import BeautifulSoup
from os import walk
from os import makedirs
from os import path


def parse_with_selector(folder_in, filename_in, selector):
    with open('./raw_data/html/in/' + folder_in + '/' + filename_in, "rb") as file:
        html = file.read()
        soup = BeautifulSoup(html)
        divs = selector(soup)

    # create the output folder if it does not exist.
    if not path.exists('./raw_data/html/out/' + folder_in):
        makedirs('./raw_data/html/out/' + folder_in)
    with open('./raw_data/html/out/' + folder_in + '/' + filename_in + '.txt', 'w', encoding="utf8", newline='') as file_out:
        for div in divs:
            if div.string is not None and div.string.rstrip() != '':
                stc = curate_sentence(str(div.string)).rstrip()
                if stc != '':
                    file_out.write(stc + '\n')


def parse_file(folder, filename):
    folder_out = './html/out/'
    with open(folder + filename, "rb") as file:
        html = file.read()
        soup = BeautifulSoup(html)
        thepost = soup.find_all('div', attrs={'dir': 'rtl'})

    divs = thepost[0].find_all('div')
    with open(folder_out + filename + '.txt', 'w', encoding="utf8", newline='') as file_out:
        for div in divs:
            # if len(div.children) == 1 and div.children[0] is NavigableString:
            if div.string is not None and div.string.rstrip() != '':
                stc = curate_sentence(str(div.string)).rstrip()
                if stc != '':
                    # print(stc)
                    file_out.write(stc)


def curate_sentence(stc):
    stc = ' '.join(stc.split(',,,,'))
    stc = ' '.join(stc.split(',,,'))
    stc = ' '.join(stc.split(',,'))
    stc = '\n'.join(stc.split(','))
    stc = ' '.join(stc.split('....'))
    stc = ' '.join(stc.split('...'))
    stc = ' '.join(stc.split('..'))
    stc = 'ا'.join(stc.split('ااااا'))
    stc = 'ا'.join(stc.split('اااا'))
    stc = 'ا'.join(stc.split('ااا'))
    stc = 'ا'.join(stc.split('اا'))
    stc = ' '.join(stc.split('^'))

    stc = 'ههه'.join(stc.split('هههههه'))
    stc = 'ههه'.join(stc.split('ههههه'))
    stc = 'ههه'.join(stc.split('هههه'))

    stc = '\n'.join(stc.split('  .'))


    # todo remove letters that are too repetitive.

    # todo remove non arabic words, especially the emojies.

    chars = "<>=+-*/#\"'&%؟;:"
    for char in chars:
        stc = ' '.join(stc.split(char))

    return stc


def parse_all_files(folders, selector, overwrite=False):
    filenames = []
    for folder in folders:
        for _, _, files in walk('./raw_data/html/in/' + folder):
            for file in files:
                filenames = filenames + [(folder, file)]

    for idx, tuplet in enumerate(filenames):
        if idx % 100 == 0:
            print('%s/%s' % (idx, len(filenames)))
        if overwrite:
            parse_with_selector(tuplet[0], tuplet[1], selector)

    return filenames


def concatenate_files(folders):
    filenames = []
    for folder in folders:
        for _, _, files in walk("./raw_data/html/out/" + folder):
            for file in files:
                filenames = filenames + ["./raw_data/html/out/" + folder + "/" + file]

    large_size = 0  # current file size
    not_large_size = 0  # current file size
    max_size = 20 * 1024 * 1024  # ~20Mb
    large_size_counter = 1
    not_large_size_counter = 1

    large_sentences_file = open('./raw_data/html/out/merged/large_sentences_1.txt',
                                'a', encoding="utf8", newline='')
    not_large_sentences_file = open('./raw_data/html/out/merged/not_large_sentences_1.txt',
                                    'a', encoding="utf8", newline='')
    for idx, file in enumerate(filenames):
        with open(file, "rb") as in_file:
            if large_size > max_size:
                large_size = 0
                large_size_counter = large_size_counter + 1
                if large_sentences_file is not None:
                    large_sentences_file.close()
                large_sentences_file = open('./raw_data/html/out/merged/large_sentences_%s.txt' % large_size_counter,
                                            'a', encoding="utf8", newline='')

            if not_large_size > max_size:
                not_large_size = 0
                not_large_size_counter = not_large_size_counter + 1
                if not_large_sentences_file is not None:
                    not_large_sentences_file.close()
                not_large_sentences_file = open('./raw_data/html/out/merged/not_large_sentences_%s.txt'
                                                % not_large_size_counter,
                                                'a', encoding="utf8", newline='')
            print("%s/%s - %s" % (idx, len(filenames), file))
            # print(large_size, not_large_size)
            for line in in_file:
                line = line.rstrip()
                line = line.decode('utf8')
                if len(bytes(line + '\n', "utf8")) > 1:
                    # print('l size:', len(bytes(line + '\n', "utf8")), line)
                    if len(line.split()) > 20:
                        large_sentences_file.write(line + '\n')
                        large_size = large_size + len((line + '\n').encode("utf-8"))
                    else:
                        not_large_sentences_file.write(line + '\n')
                        not_large_size = not_large_size + len(bytes(line + '\n', "utf8"))
    large_sentences_file.close()
    not_large_sentences_file.close()


def _selector(html):
    thepost = html.find_all('div', attrs={'dir': 'rtl'})
    return thepost[0].find_all('div')


if __name__ == '__main__':
    # parse files.
    folders = ['some_html_file']
    parse_all_files(folders, lambda s: _selector(s), overwrite=False)

    # merge all files
    all_folders = ['some_output_folder']

    # for folder in all_folders:
    concatenate_files(all_folders)
