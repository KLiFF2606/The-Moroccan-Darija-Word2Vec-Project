"""
Todo Comment this file.
"""


def clean_line(line, strict_arabic=True, replace_char=''):
    out = ''
    for word in line.split():
        if str(word).isdecimal() or str(word).isdigit():
            out = out.rstrip() + ' NUMBER '
        else:
            for char in list(word):
                u = ('' + str(char)).encode('utf-8')
                # print(char, '-', len(u))
                if strict_arabic:
                    if len(u) == 2:
                        out = out + str(char)
                    else:
                        out = out + replace_char
                else:
                    # this does not really anything.
                    out = out + char
            out = out.rstrip() + ' '
    return out.rstrip()


def chat2csv(filename, output):
    """
    line is in this format: 05/03/2018, 16:19 - <nickname/phone>: <message>
    :param filename:
    :param output:
    :return:
    """
    with open(filename, 'rb') as f_in, \
            open(output, 'w', encoding="utf8", newline='') as f_out:
        for line in f_in:
            # remove invisible chars
            line = line.rstrip()
            # remove the first chars
            if len(line) > 5 and (line[2] == line[5] == 47):
                line = line[20:]
                try:
                    line = line.decode('utf8')
                    # get everything after the first ':'
                    line = line[line.index(':') + 2:]
                except UnicodeDecodeError as ude:
                    pass
                except ValueError as ve:
                    pass

                if str(line) != '<Media omitted>' and str(line) != '<Médias omis>':
                    orig = line
                    line = clean_line(line, replace_char='')
                    if len(line) != 0:
                        f_out.write(str(line) + '\n')
            else:
                # a new line inside a sentence.
                # ignore
                continue


if __name__ == '__main__':
    chat2csv('./raw_data/WhatsApp Chats/WhatsApp Chat 1.txt', './data/data_output_1.txt')
