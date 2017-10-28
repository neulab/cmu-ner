import sys


def run_program():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program takes two arguments: the output file and the DARPA unlabelled test file')
    with open(sys.argv[1], 'r') as input_file:
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

    with open(sys.argv[2], 'r') as input_file:
        lines = input_file.readlines()
    assert len(output_lines) == len(lines)
    ctr = -1
    for line in lines:
        if len(line) > 2:
            ctr += 1
            line_split = line.strip().split()
            assert len(line_split) == 10
            print '\t'.join(line_split) + '\t' + tags[ctr]
        else:
            print ''
    assert ctr + 1 == len(tags)


if __name__ == "__main__":
    run_program()
