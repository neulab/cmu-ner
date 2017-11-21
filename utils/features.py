import utils.segnerfts as segnerfts
import codecs


def get_feature_w(lang, w):
    return segnerfts.extractIndicatorFeatures(lang, w)


def get_brown_cluster(path):
    bc_dict = dict()
    with codecs.open(path, "r", "utf-8") as fin:
        for line in fin:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                word = fields[1]
                binary_string = fields[0]
                bc_dict[word] = int(binary_string, 2)
    return bc_dict