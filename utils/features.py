import utils.segnerfts as segnerfts
import codecs


def get_feature_w(lang, sent):
    return segnerfts.extract(lang, sent)


def get_brown_cluster(path):
    bc_dict = dict()
    linear_map = dict()
    with codecs.open(path, "r", "utf-8") as fin:
        for line in fin:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                word = fields[1]
                binary_string = fields[0]
                bid = int(binary_string, 2)
                if bid not in linear_map:
                    linear_map[bid] = len(linear_map)
                bc_dict[word] = linear_map[bid]
    return bc_dict