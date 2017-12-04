import sys
import codecs
from random import shuffle


def split(path, write_to, split_num):
    tot_data = []

    with codecs.open(path, "r", "utf-8") as fin:
        one_sent = []
        for line in fin:
            if line.strip() == "":
                if len(one_sent) > 0:
                    tot_data.append(one_sent)
                one_sent = []
            else:
                one_sent.append(line.strip())
        if len(one_sent) > 0:
            tot_data.append(one_sent)

    shuffle(tot_data)

    divs = len(tot_data) / split_num
    splits = range(0, len(tot_data), divs)
    splits[-1] = len(tot_data)
    for i in range(split_num):
        with codecs.open(write_to + "cp3_train_ens_" + str(i) + ".conll", "w", "utf-8") as fout:
            for j in range(splits[i], splits[i+1]):
                for line in tot_data[j]:
                    fout.write(line + "\n")
                fout.write("\n")

if __name__ == "__main__":
    # Usage: python split_train_ensemble.py ../datasets/cp3/oromo/cp3_train.conll ../datasets/cp3/oromo/ 5
    fname = sys.argv[1]
    write_to_folder = sys.argv[2]
    split_num = int(sys.argv[3])

    split(fname, write_to_folder, split_num)