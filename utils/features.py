from utils.segnerfts import segnerfts
import codecs


def get_feature_sent(lang, sent, args):
    if args.use_gazatter and args.use_morph:
        return segnerfts.extract(lang, sent)
    elif args.use_gazatter:
        return segnerfts.extract_type_token_gaz(lang, sent)
    elif args.use_morph:
        return segnerfts.extract_type_token_morph(lang, sent)
    else:
        return segnerfts.extract_type_token_level(lang, sent)


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