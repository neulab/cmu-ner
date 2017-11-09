import sys
import argparse
import codecs

def run_program(input, output, setEconll):
    reload(sys)	
    sys.setdefaultencoding('utf-8')
    if input is not None and setEconll is not None:
        with codecs.open(input, 'r',encoding='utf-8', errors='ignore') as input_file:
            lines = input_file.readlines()
        tags = []
        for i, line in enumerate(lines):
            if len(line) >= 2:
                line_split = line.strip().split()
                sys.stderr.write('line: ' + line.strip() + '\n')
                sys.stderr.flush()
                assert len(line_split) == 4
                tags.append(line_split[-1])

        output_lines = lines

        with codecs.open(setEconll, 'r',encoding='utf-8', errors='ignore') as input_file:
            lines = input_file.readlines()
        assert len(output_lines) == len(lines)
        with codecs.open(output,'w',encoding='utf-8') as output_file:
            ctr = -1
            for line in lines:
                if len(line) > 2:
                    ctr += 1
                    line_split = line.strip().split()
                    assert len(line_split) == 10
                    print '\t'.join(line_split) + '\t' + tags[ctr]
                    output_file.write('\t'.join(line_split) + '\t' + tags[ctr] +"\n")
                else:
                    print ""
                    output_file.write("\n")
            assert ctr + 1 == len(tags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--setEconll", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    #run_program(args)
