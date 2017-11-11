import utils.segnerfts


def get_feature_w(lang, w):
    return utils.segnerfts.extractIndicatorFeatures(lang, w)
