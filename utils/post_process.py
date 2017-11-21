import codecs

def post_process(path_to_prediction, path_to_psm=None):
    with codecs.open(path_to_prediction, "r", "utf-8") as fin:
        for line in fin:
            pass