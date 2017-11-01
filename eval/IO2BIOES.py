import sys

def transform(ifile, ofile):
	with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
		sents = []
		sent = []
		for line in reader:
			line = line.strip()
			if len(line) == 0:
				sents.append(sent)
				sent = []
				continue

			sent.append(line)
		if len(sent) > 0:
			sents.append(sent)

		for sent in sents:
			length = len(sent)
			labels = []
			for line in sent:
				tokens = line.split()
				label = tokens[-1]
				labels.append(label)
			
			# print "%d %d" % (length, len(labels))

			for i in range(length):
				tokens = sent[i].split()
				label = labels[i]
				new_label = label
				if label != 'O':
					if label.startswith('B-'):
						if i + 1 == length or not labels[i + 1].startswith('I-'):
							new_label = 'S-' + label[2:]
					elif label.startswith('I-'):
						if i + 1 == length or not labels[i + 1].startswith('I-'):
							new_label = 'E-' + label[2:]
				writer.write(" ".join(tokens[:-1]) + " " + new_label)
				writer.write('\n')
			writer.write('\n')


if __name__ == '__main__':
	transform('../datasets/english/eng.train.bio.conll', '../datasets/english/eng.train.bioes.conll')
	transform('../datasets/english/eng.dev.bio.conll',   '../datasets/english/eng.dev.bioes.conll')
	transform('../datasets/english/eng.test.bio.conll',  '../datasets/english/eng.test.bioes.conll')
