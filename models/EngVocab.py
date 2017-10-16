import numpy as np
import codecs


def GenerateEmbeddingsAndVocab(inputFileName,word_to_id,char_to_id,word_emb):

    for line in codecs.open(inputFileName, 'r', 'utf-8', errors='ignore'):
        items = line.rstrip().split()
        if items[0] not in word_to_id:
            word_to_id[items[0]] = len(word_to_id)
            word_emb.append(np.asarray(items[1:]).astype(float))
            for c in items[0]:
                if c not in char_to_id:
                    char_to_id[c] = len(char_to_id)
        else:
            word_emb[word_to_id[items[0]]] = np.asarray(items[1:]).astype(float)


    return word_emb, word_to_id,char_to_id

def GenerateCombinedEmbeddingsAndVocab(lang1EmbFile, lang2EmbFile):
    word_emb = []
    vocab = {}
    vocab["<eos>"] = 0
    word_emb.append(np.zeros(50).astype(float))

    for line in codecs.open(lang1EmbFile, 'r', 'utf-8', errors='ignore'):
        items = line.rstrip().split()
        vocab[items[0]] = len(vocab)
        word_emb.append(np.asarray(items[1:]).astype(float))

    for line in codecs.open(lang2EmbFile, 'r', 'utf-8', errors='ignore'):
        items = line.rstrip().split()
        vocab[items[0]] = len(vocab)
        word_emb.append(np.asarray(items[1:]).astype(float))

    vocab["<unk>"] = len(vocab) + 1
    word_emb.append(np.zeros(50).astype(float))

    return word_emb, vocab

#word_emb,vocab = GenerateCombinedEmbeddingsAndVocab("/Users/aditichaudhary/Documents/CMU/Lorelei/cross-lingual/ti_emb","/Users/aditichaudhary/Documents/CMU/Lorelei/glove.6B/glove.6B.50d.txt")

