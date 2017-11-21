import os
import codecs
import xml.etree.ElementTree as ET
import sys

def extract_authors(dir_name, output_fname):
    author_set = set()
    for fname in os.listdir(dir_name):
        fin_name = os.path.join(dir_name, fname)
        if os.path.isfile(fin_name):
            fs = fname.split('_')
            if fs[1] != "WL":
                continue
            print fname
            tree = ET.parse(fin_name)
            root = tree.getroot()
            # elems = root.findall(".//*[@type='post']/[@name='author']")
            elems = root.findall(".//*[@type='post']/attribute")
            for elem in elems:
                if elem.get('name') == u'author':
                    author = elem.get(u'value')
                    author_set.add(author)

    with codecs.open(output_fname, "w", "utf-8") as fout:
        for elem in author_set:
            fout.write(elem + '\n')

if __name__ == "__main__":
    dname = sys.argv[1]
    fout_name = sys.argv[2]
    extract_authors(dname, fout_name)