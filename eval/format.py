import sys


def format(ifile, ofile):
    with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
        i = 1
        for line in reader:
            line = line.strip()
            if len(line) == 0:
                i = 1
                writer.write('\n')
            else:
                writer.write('%d %s\n' % (i, line))
                i += 1


if __name__ == '__main__':
    format('eng.train', 'eng.train.conll')
    format('eng.dev', 'eng.dev.conll')
    format('eng.test', 'eng.test.conll')
