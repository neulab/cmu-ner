import sys
import codecs
import argparse


def print_entities(fout,entities, curr_docum, curr_anot):
    print 'CMU_NER_LOREAL_CP1_TB_GS' + '\t' + curr_docum + '-ann-' + str(curr_anot) + '\t' + ' '.join(
        entities[0]) + '\t' + curr_docum + ':' + str(entities[2]) + '-' + str(entities[3]) + '\t' + 'NIL' + '\t' + \
          entities[1] + '\t' + 'NAM' + '\t' + '1.0'
    fout.write('CMU_NER_LOREAL_CP1_TB_GS' + '\t' + curr_docum + '-ann-' + str(curr_anot) + '\t' + ' '.join(
        entities[0]) + '\t' + curr_docum + ':' + str(entities[2]) + '-' + str(entities[3]) + '\t' + 'NIL' + '\t' + \
          entities[1] + '\t' + 'NAM' + '\t' + '1.0' + "\n")


def run_program(args):
    reload(sys)
    sys.setdefaultencoding('utf-8')
    if args.input is not None and args.output is not None:
        with codecs.open(args.input, encoding='utf-8', mode='r') as input_file:
            lines = input_file.readlines()

        entities = [[], None, -1, -1]
        in_entity = False
        curr_docum = None
        curr_anot = 1
        fout = codecs.open(args.output,'w')
        for i, line in enumerate(lines):
            if len(line) > 2:
                sys.stderr.write('Line number: ' + str(i + 1) + '\n')
                sys.stderr.flush()
                line_split = line.strip().split()
                if curr_docum != line_split[3]:
                    curr_docum = line_split[3]
                    curr_anot = 1
                    # print ''
                if len(line_split) != 11:
                    sys.stderr.write(line)
                    sys.stderr.write('Error in line: ' + str(i + 1) + '\n')
                    assert len(line_split) == 11
                if line_split[-1][0] == 'B':
                    if in_entity:
                        print_entities(fout,entities, curr_docum, curr_anot)
                        # restart
                        entities[0] = []
                        entities[1] = None
                        entities[2] = -1
                        entities[3] = -1
                        curr_anot += 1
                        in_entity = False
                    else:
                        assert len(entities[0]) == 0 and entities[1] is None and entities[2] == -1 and entities[3] == -1
                    assert not (in_entity)
                    in_entity = True
                    assert line_split[-1][1] == '-'
                    entities[0].append(line_split[0])
                    entities[1] = ''.join(line_split[-1][2:])
                    entities[2] = int(line_split[-5])
                    entities[3] = int(line_split[-4])
                elif line_split[-1][0] == 'I':
                    sys.stderr.write('line num: ' + str(i + 1) + '\n')
                    assert in_entity and len(entities[0]) > 0 and not (entities[0] is None) and ''.join(
                        line_split[-1][2:]) == entities[1] and entities[2] >= 0 and entities[3] >= 0
                    entities[0].append(line_split[0])
                    assert entities[2] >= 0
                    assert int(line_split[-4]) > entities[3]
                    entities[3] = int(line_split[-4])
                elif line_split[-1][0] == 'O':
                    if in_entity:
                        print_entities(fout,entities, curr_docum, curr_anot)
                        entities[0] = []
                        entities[1] = None
                        entities[2] = -1
                        entities[3] = -1
                        curr_anot += 1
                        in_entity = False
            else:
                if in_entity:
                    sys.stderr.write('We are in an entity and met sentence boundary, line: ' + str(i + 1) + '\n')
                    print_entities(fout,entities, curr_docum, curr_anot)
                    entities[0] = []
                    entities[1] = None
                    entities[2] = -1
                    entities[3] = -1
                    curr_anot += 1
                    in_entity = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_program(args)
