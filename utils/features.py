import utils.segnerfts as segnerfts


def get_feature_w(lang, w):
    return segnerfts.extractIndicatorFeatures(lang, w)
