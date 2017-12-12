#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from io import open
import epitran
import sys, json, glob, os, math
from copy import deepcopy
from morpar_orm import *
import nltk
from nltk.probability import *
import cPickle as pickle
from collections import defaultdict
import os.path

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')
from nltk.corpus import brown
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache
    
import lxml.etree as ET
from collections import defaultdict
def log_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# MEMOIZATION
channelIndexDict = {}
channels = ["breakdown", "lemma", "gloss", "citation", "natural"]

for index, channel in enumerate(channels):
	channelIndexDict[channel] = index


#with open("orm_common_dict.pkl", "rb") as handle:
#	commonParseDict = pickle.load(handle)

# NORMALIZATION


import re

PATTERNS = [
    (r'([aeiou])(\1)', r'\1'),
    (r'(b|c|ch|d|dh|f|g|h|j|k|l|m|n|ny|p|ph|q|r|s|sh|t|v|w|x|y|z)\1', r'\1'),
    (r'ph', r'p'),
    (r'q', r'k'),
    (r'x', r't'),
    (r'c([^h]|\b)', r'ch\1'),
    (r'ai', r'ayi'),
]


def normalize(text):
    if all([x.isupper() for x in text]):
        return text
    cap = True if text[0].isupper() and all([x.islower() for x in text[1:]]) else False
    text = text.lower()
    for pattern, repl in PATTERNS:
        text = re.sub(pattern, repl, text)
    if cap:
        return text.capitalize()
    else:
        return text


######################################
#
# LOOKUP PARSER
#
# This section defines a custom parser
# that looks up a word in the LDC Amharic
# dictionary.
#
######################################
setSWords = []
setSList = open("/home/data/LoReHLT17/internal/Lexicons/orm_lexicon/setS_wordlist.txt", "r")
for line in setSList:
	setSWords.append(line.strip())

def get_freq_dist():
    freq = FreqDist()
    freq.update(brown.words())
    freq.update(setSWords)
    return freq
epi = epitran.Epitran("orm-Latn")
g2p = epi.transliterate

def stripFinalVowel(string):
	if string[-2:] in ["aa", "ee", "ii", "oo", "uu"]:
		return string[:-2]
	elif string[-1] in ["a", "e", "i", "o", "u"]:
		return string[:-1]
	else:
		return string

def get_dictionary(dict_filenames):
    l1_to_l2 = defaultdict(list)
    
    for dict_filename in dict_filenames:
	if os.path.isfile(dict_filename):
    		with open(dict_filename, "r", encoding="utf-8") as fin:
			for line in fin.readlines():
            			parts = line.strip().split("\t")
            			if len(parts) < 2:
                			print("WARNING: Insufficient parts for line %s" % line)
                			continue
            			definition = parts[0]
            			word = parts[1]
				norm = normalize(word)
            			#ipa = g2p(word)
            			#ipa = word     
	    			l1_to_l2[norm.lower()].append((word,definition))
	  	  		unVoweled = stripFinalVowel(norm)
	    			if unVoweled != norm:
					l1_to_l2[unVoweled].append((word,definition))
 	    		#	normalized = normalize(word)
	    		#	if normalized != word and normalized != unVoweled:
			#		l1_to_l2[normalized].append((word,definition))
	else:
		print("WARNING: Missing file " + dict_filename) 
	   
    return l1_to_l2

gazetteer = open("/home/data/LoReHLT17/internal/Morph/Orm/v4/orm_gaz.txt", "r")
gazDict = defaultdict(list)

for line in gazetteer:
	parts = line.strip().split("\t")
	orm  = parts[1].split()
	for ormWord in orm:
		if ormWord[-1] == "," or ormWord[-1] == ";":
			ormWord = ormWord[:-1]
		gazDict[ormWord].append(parts[0])

knightFile = open("/home/data/LoReHLT17/internal/Lexicons/orm_lexicon/orm_knight_lexicon.tsv", "r")

for line in knightFile:
	parts = line.strip().split("\t")
	orm  = parts[1].split()
	for ormWord in orm:
		if ormWord[-1] == "," or ormWord[-1] == ";":
			ormWord = ormWord[:-1]
		gazDict[ormWord].append(parts[0])



class Lookup(Parser):
    def __init__(self, dictionaryList, channel=None, output_channel=None):
        self.dictionary = get_dictionary(dictionaryList)
        self.channel = channel
        self.output_channel = output_channel
        self.freqDist = get_freq_dist()
        self.engWords = self.freqDist.N()
    
    @lru_cache(maxsize=1000)
    def __call__(self, input, input_channel=None, leftward=False):
        results = set()
        
        text = input[input_channel.name]
        text = text.strip()
        
        output = HashableDict()
        remnant = HashableDict({input_channel.name:input_channel.typ()})
        
        for channel in self.channel:
            if channel == input_channel:
                continue
            output[channel.name] = channel.typ(text)    
        low = normalize(text).lower()
#        if text in self.dictionary:
#            for pair in self.dictionary[text]:
#                definition = pair[1]
#		lemma = pair[0]
#		cost = 0
#                for word in definition.split():
#                    word = word.lower()
#                    if word not in self.freqDist:
#                        cost += 15
#                    else:
#                        log_freq = -math.log(self.freqDist[word] * 1.0 / self.engWords)
#                        cost += math.floor(log_freq)
#                output2 = deepcopy(output)
#                for channel in self.output_channel:
#                    output2[channel.name] = channel.typ(definition)
#                output2["lemma"] = lemma
#		output2[Cost.name] = Cost.typ("X" * int(cost))
#                results.add((output2,remnant))
        if low in self.dictionary:
            for pair in self.dictionary[low]:
                definition = pair[1]
		lemma = pair[0]
		cost = 0
                for word in definition.split():
                    word = word.lower()
                    if word not in self.freqDist:
                        cost += 15
                    else:
                        log_freq = -math.log(self.freqDist[word] * 1.0 / self.engWords)
                        cost += math.floor(log_freq)
                output2 = deepcopy(output)
                for channel in self.output_channel:
                    output2[channel.name] = channel.typ(definition)
                output2["lemma"] = lemma
		output2[Cost.name] = Cost.typ("X" * int(cost))
                results.add((output2,remnant))
        if text in gazDict:
		for english in gazDict[text]:
			cost = 0
		
			for word in english.split():
				word = word.lower()
				if word not in self.freqDist:
					cost += 15
				else:
					log_freq = -math.log(self.freqDist[word] * 1.0 / self.engWords)
					cost += math.floor(log_freq)
			output2 = deepcopy(output)
			for channel in self.output_channel:
				output2[channel.name] = channel.typ(english)
			output2[Cost.name] = Cost.typ("X" * int(cost))
			results.add((output2,remnant))
#	if len(results) == 0 and normalize(low) in self.dictionary:
#	     newText = normalize(low)
#	     for pair in self.dictionary[newText]:
#               definition = pair[1]
#		lemma = pair[0]
#		cost = 0
#                for word in definition.split():
#                    word = word.lower()
#                    if word not in self.freqDist:
#                        cost += 10
#                    else:
#                        log_freq = -math.log(self.freqDist[word] * 1.0 / self.engWords)
#                        cost += math.floor(log_freq)
#                output2 = deepcopy(output)
#                for channel in self.output_channel:
#                    output2[channel.name] = channel.typ(definition)
#                output2["lemma"] = lemma
#		output2[Cost.name] = Cost.typ("X" * int(cost))
#                results.add((output2,remnant))	
	if len(results) == 0:
            #print("didn't find it: %s" % text)
            output2 = deepcopy(output)
            for channel in self.output_channel:
                output2[channel.name] = channel.typ(text.replace(" ",""))
            cost = "X" * (50 + len(text))
            output2[Cost.name] = Cost.typ(cost)
            results.add((output2,remnant))
            
        return results

###############################
#
# MORPHOLOGICAL GRAMMAR
#
###############################
Cost = Concatenated("cost")
Nat = Concatenated("natural")
#PARSER      = Lookup("orm_lexicon.txt", Tex/Mor/Lem, Glo/Cit/Nat)
LEMMA = Lookup(("/home/data/LoReHLT17/internal/Morph/Orm/v4/orm_lexicon.txt", "/home/data/LoReHLT17/internal/Morph/Orm/v4/orm_lexicon_wikibooks.txt", "/home/data/LoReHLT17/internal/Morph/Orm/v4/lexicon_supplement.txt"), Tex/Mor/Lem, Glo/Cit/Nat)

##############################
#
# CONVENIENCE FUNCTIONS
#
##############################

# http start, html end

@lru_cache(maxsize=1000)
def parse(word, representation_name="lemma"):
    #ipa = g2p(word)
    
 #   if word in commonParseDict:
#	return [unicode(item) for item in commonParseDict[word][0][channelIndexDict[representation_name]]]

    parses = PARSER.parse(normalize(word))
    #parses = PARSER.parse(ipa)
    if not parses:
        print("Warning: cannot parse %s (%s)" % (word, ipa))
        parses = [{representation_name:ipa,"cost":""}]
    parses.sort(key=lambda x:len(x["cost"]) if "cost" in x else 0)
    #print([x[representation_name] for x in parses])
    return [unicode(x[representation_name]) for x in parses]
@lru_cache(maxsize=1000)
def best_parse(word, representation_name="lemma"):
    #if word in commonParseDict:
#	return unicode(commonParseDict[word][1][channelIndexDict[representation_name]])    

    return unicode(parse(word, representation_name)[0])



VV = "(aa|ee|ii|oo|uu)"
V = "(a|e|i|o|u)"
C = "(b|c|ch|d|dh|f|g|h|j|k|l|m|n|ny|p|ph|q|r|s|sh|t|v|w|x|y|z)"
CC = C + C

#LEMMA = Guess(Tex/Mor/Lem/Glo)

#VERB = PRE_VERB + VERB_ROOT + VOICE_EXTENSION + PERSON + PLURAL_TENSE + CASE

# Morpheme-boundary phonological changes
# Open things: Some assimilations to double consonants (tt, nn) may variably be written as single (t, n)?
NTexMor = lambda x : Tex/Mor(x) | After("r", Tex) + Tex("r" + x[1:]) + Mor(x) | (After("l",Tex) + Tex("l" + x[1:]) + Mor(x)) | Truncate("t", Tex) + Tex("n") + Tex/Mor(x) | Truncate("x", Tex) + Tex("n") + Tex/Mor(x) | Truncate("d", Tex) + Tex("n") + Tex/Mor(x) | Truncate("dh", Tex) + Tex("n") + Tex/Mor(x) | Truncate("s", Tex) + Tex("f") + Tex/Mor(x) 
TTexMor = lambda x: Tex/Mor(x) | After("(b|g|d)", Tex) + Tex("d" + x[1:]) + Mor(x) | After("(x|q)", Tex) + Tex("x" + x[1:]) + Mor(x) | Truncate("dh", Tex) + Tex("t") + Tex/Mor(x) | Truncate("s", Tex) + Tex("f") + Tex/Mor(x)


# Lengthening a vowel--takes the gloss as its argument
#LengthenGlo = lambda x: After("a", Tex) + Tex("a") + Glo(x) | After("e", Tex) + Tex("e") + Glo(x) | After("i", Tex) + Tex("i") + Glo(x) | After("o", Tex) + Tex("o") + Glo(x) | After("u", Tex) + Tex("u") + Glo(x)
#Lengthen = After("a", Tex) + Tex("a") | After("e", Tex) + Tex("e") | After("i", Tex) + Tex("i") | After("o", Tex) + Tex("o") | After("u", Tex) + Tex("u")
#LengthenCons = After("b", Tex) + Tex("b") | After("c", Tex) + Tex("c") | After("ch", Tex) + Tex("ch") | After("d", Tex) + Tex("d") | After("dh", Tex) + Tex("dh") | After("f", Tex) + Tex("f") | After("g", Tex) + Tex("g") | After("h", Tex) + Tex("h") | After("j", Tex) + Tex("j") | After("k", Tex) + Tex("k") | After("l", Tex) + Tex("l") | After("m", Tex) + Tex("m") | After("n", Tex) + Tex("n") | After("ny", Tex) + Tex("ny") | After("p", Tex) + Tex("p") | After("ph", Tex) + Tex("ph") | After("q", Tex) + Tex("q") | After("r", Tex) + Tex("r") | After("s", Tex) + Tex("s") | After("sh", Tex) + Tex("sh") | After("t", Tex) + Tex("t") | After("v", Tex) + Tex("v") | After("w", Tex) + Tex("w") | After("x", Tex) + Tex("x") | After("y", Tex) + Tex("y") | After("z", Tex) + Tex("z")


# Remove final vowel
#RemoveFinalVowel = NULL | Truncate("a", Tex) | Truncate("e", Tex) | Truncate("i", Tex) | Truncate("o", Tex) | Truncate("u", Tex) | Truncate("aa", Tex) | Truncate("ee", Tex) | Truncate("ii", Tex) | Truncate("oo", Tex) | Truncate("uu", Tex)


NOUN_STEM = LEMMA

NOUN_PLURAL = NULL | \
		 Tex/Mor("ota") + Glo("PL") + Nat("more than one (.*)") | \
		 TTexMor("tota") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("wan") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("en") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("an") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("le") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("yi") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("oti") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("oli") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("ot") + Glo("PL") + Nat("more than one (.*)") | \
		 TTexMor("tot") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("l") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("y") + Glo("PL") + Nat("more than one (.*)") | \
		 Tex/Mor("ol") + Glo("PL") + Nat("more than one (.*)")
		

NOUN_DEF = NULL | \
		 Tex/Mor("icha") + Glo("M.DEF") + Nat("the (.*)") | \
		 TTexMor("ticha") + Glo("M.DEF") + Nat("the (.*)") | \
		 Tex/Mor("iti") + Glo("F.DEF") + Nat("the (.*)") | \
		 TTexMor("titi") + Glo("F.DEF") + Nat("the (.*)") | \
		 Tex/Mor("ich") + Glo("M.DEF") + Nat("the (.*)") | \
		 TTexMor("tich") + Glo("M.DEF") + Nat("the (.*)") | \
		 Tex/Mor("it") + Glo("F.DEF") + Nat("the (.*)") | \
		 TTexMor("tit") + Glo("F.DEF") + Nat("the (.*)")

# -in might be a nominative case marker too
NOUN_CASE = NULL | \
		After(V, Tex) + Tex/Mor("n") + Glo("NOM") | \
		After("%s%s" % (V,C), Tex) + NTexMor("ni") + Glo("M.NOM") | \
		After("%s" % C, Tex) + TTexMor("ti") + Glo("F.NOM") | \
		After("%s" % C, Tex) + Tex/Mor("i") + Glo("NOM") | \
		After(C) + Tex/Mor("i") + Glo("GEN") + Nat("of (.*)") | \
		Tex/Mor("f") + Glo("DAT") + Nat("to (.*)") | \
		Tex/Mor("f") + Glo("DAT") + Nat("to (.*)") | \
		After(C) + Tex/Mor("if") + Glo("DAT") + Nat("to (.*)") | \
		After(V) + Tex/Mor("dha") + Glo("DAT") + Nat("to (.*)") | \
		After(V) + Tex/Mor("dhaf") + Glo("DAT") + Nat("to (.*)") | \
		TTexMor("ti") + Glo("DAT") + Nat("to (.*)") | \
		After(V) + Tex/Mor("n") + Glo("INST") + Nat("with (.*)") | \
		After(C) + Tex/Mor("in") + Glo("INST") + Nat("with (.*)") | \
		After(V) + Tex/Mor("tin") + Glo("INST") + Nat("with (.*)") | \
		After(V) + Tex/Mor("dhan") + Glo("INST") + Nat("with (.*)") | \
		TTexMor("ti") + Glo("LOC") + Nat("in (.*)") | \
		After(VV) + Tex/Mor("dha") + Glo("ABL") + Nat("from (.*)") | \
		After(C) + Tex/Mor("i") + Glo("ABL") + Nat("from (.*)") | \
		TTexMor("ti") + Glo("ABL") + Nat("from (.*)") | \
		After(V, Tex) + Tex/Mor("n") + Glo("NOM") | \
		After("%s%s" % (V,C), Tex) + NTexMor("n") + Glo("M.NOM") | \
		After("%s" % C, Tex) + TTexMor("t") + Glo("F.NOM") | \
		After(V) + Tex/Mor("dh") + Glo("DAT") + Nat("to (.*)") | \
		TTexMor("t") + Glo("DAT") + Nat("to (.*)") | \
		TTexMor("t") + Glo("LOC") + Nat("in (.*)") | \
		After(VV) + Tex/Mor("dh") + Glo("ABL") + Nat("from (.*)") | \
		TTexMor("t") + Glo("ABL") + Nat("from (.*)")	
		
LENGTHENER = NULL | TTexMor("ti") + Glo("LENG")

EMPHASIS = NULL | \
		Tex/Mor("tu") + Glo("EMPH")

ONLY = NULL |  Tex/Mor("uma") + Glo("ONLY") + Nat("only (.*)")

EVEN = NULL | Tex/Mor("le") + Glo("EVEN") + Nat("even (.*)")

COORD = NULL | \
	Tex/Mor("f") + Glo("AND") + Nat("(.*) and") | \
	Tex/Mor("fi") + Glo("AND") + Nat("(.*) and") | \
	Tex/Mor("s") + Glo("AS_WELL") + Nat("as well as (.*)")

MYSTERY = NULL | \
		Tex/Mor("fa") + Glo("UNKNOWN")

FIRST_PERSON = NULL | Tex/Mor("n") + Glo("1SG")

#TITLE = NULL | Tex/Mor("ob.") + Glo("MR") + Nat("Mr. (.*)")

NOUN = NOUN_STEM + NOUN_PLURAL + NOUN_DEF + LENGTHENER + ONLY + EMPHASIS + NOUN_CASE + EMPHASIS + EVEN + COORD + MYSTERY + FIRST_PERSON



VERB_STEM = LEMMA + Truncate("u", Tex) | LEMMA + Truncate("chu", Tex) + Tex/Mor("dh") | LEMMA + Truncate("chuu", Tex) + Tex/Mor("t")

VOWEL_EPENTHESIS = NULL | After(CC, Tex) + Tex("i") | After(CC, Tex) + Tex("a")  

ASP_AGR = Tex/Mor("e") + Glo("1SG.PST.AFF") + Nat("I did (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("te") + Glo("2SG.PST.AFF") + Nat("you-sg did (.*)") | \
		Tex/Mor("e") + Glo("3SGM.PST.AFF") + Nat("he did (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("te") + Glo("3SGF.PST.AFF") + Nat("she did (.*)") | \
		VOWEL_EPENTHESIS + NTexMor("ne") + Glo("1PL.PST.AFF") + Nat("we did (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tani") + Glo("2PL.PST.AFF") + Nat("you-pl did (.*)") | \
		Tex/Mor("ani") + Glo("3PL.PST.AFF") + Nat("they did (.*)") | \
		VOWEL_EPENTHESIS + NTexMor("ne") + Glo("PST.NEG") + Nat("did not (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("ta") + Glo("2SG.PRS.AFF") + Nat("you-sg (.*)") | \
		Tex/Mor("a") + Glo("PRS.AFF") + Nat("he (.*)s") | \
		VOWEL_EPENTHESIS + TTexMor("ti") + Glo("3SGF.PRS.AFF") + Nat("she (.*)s") | \
		VOWEL_EPENTHESIS + NTexMor("na") + Glo("1PL.PRS.AFF") + Nat("we (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tu") + Glo("2PL.PRS.AFF") + Nat("you-pl (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tani") + Glo("2PL.PRS.AFF") + Nat("you-pl (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tan") + Glo("2PL.PRS.AFF") + Nat("you-pl (.*)") | \
		Tex/Mor("ani") + Glo("3PL.PRS.AFF") + Nat("they (.*)") | \
		Tex/Mor("an") + Glo("3PL.PRS.AFF") + Nat("they (.*)") | \
		Tex/Mor("u") + Glo("1SG.PRS.NEG") + Nat("I do not (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tu") + Glo("2SG.PRS.NEG") + Nat("you-sg do not (.*)") | \
		Tex/Mor("u") + Glo("3SGM.PRS.NEG") + Nat("he does not (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tu") + Glo("3SGF.PRS.NEG") + Nat("she does not (.*)") | \
		VOWEL_EPENTHESIS + NTexMor("nu") + Glo("1PL.PRS.NEG") + Nat("we do not (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tan") + Glo("2PL.PRS.NEG") + Nat("you-pl do  not (.*)") | \
		Tex/Mor("an") + Glo("3PL.PRS.NEG") + Nat("they do not (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tu") + Glo("2SG.PRS.AFF") + Nat("you-sg (.*)") | \
		Tex/Mor("u") + Glo("PRS.AFF") + Nat("he (.*)s") | \
		VOWEL_EPENTHESIS + TTexMor("tu") + Glo("3SGF.PRS.AFF") + Nat("she (.*)s") | \
		VOWEL_EPENTHESIS + NTexMor("nu") + Glo("1PL.PRS.AFF") + Nat("we (.*)") | \
		VOWEL_EPENTHESIS + TTexMor("tani") + Glo("2PL.PRS.AFF") + Nat("you-pl (.*)") | \
		Tex/Mor("ani") + Glo("3PL.PRS.AFF") + Nat("they (.*)") | \
		VOWEL_EPENTHESIS + NTexMor("ne") + Glo("PRS.NEG") + Nat("not (.*)") | \
		Tex/Mor("i") + Glo("2SG.JUSS.AFF") + Nat("you-sg, (.*)!") | \
		Tex/Mor("u") + Glo("JUSS.AFF") + Nat("let him (.*)!") | \
		VOWEL_EPENTHESIS + TTexMor("tu") + Glo("3SGF.JUSS.AFF") + Nat("let her (.*)!") | \
		VOWEL_EPENTHESIS + NTexMor("nu") + Glo("1PL.JUSS.AFF") + Nat("let us (.*)!") | \
		Tex/Mor("a") + Glo("2PL.JUSS.AFF") + Nat("you-pl, (.*)!") | \
		Tex/Mor("anu") + Glo("3PL.JUSS.AFF") + Nat("let them (.*)!") | \
		Tex/Mor("in") + Glo("2SG.JUSS.NEG") + Nat("you-sg, don't (.*)!") | \
		Tex/Mor("in") + Glo("JUSS.NEG") + Nat("let him not (.*)!") | \
		Tex/Mor("ina") + Glo("2PL.JUSS.NEG") + Nat("you-pl, don't (.*)!") | \
		Tex/Mor("ina") + Glo("2PL.JUSS.NEG") + Nat("you-pl, don't (.*)!")
		


INFINITIVE = NULL | \
		Tex/Mor("u") + Glo("INF") + Nat("to (.*)")



VERB = VERB_STEM + ASP_AGR | VERB_STEM + INFINITIVE




PARSER = NOUN | VERB

#words = ["taatuun", "qilleensi", "jaballi", "afaan", "loltoonni", "namichi", "waantooti", "namichaa", "Caaltuu", "afaanii", "namichaa", "intalaaf", "sareef", "baruuf", "bishaaniif", "sareedhaa", "sareedhaaf", "Caaltuutti"]
pairs = []
pairs = [("taatuun","taatuu-NOM"), ("qilleensi","qilleensa-NOM"), ("jaballi","jabala-M.NOM"), ("afaan","afaan-NOM"), ("loltoonni","loltoota-M.NOM"), ("namichi","namicha-NOM"), ("waantooti","waantoota-NOM"), ("namichaa","namicha-GEN"), ("Caaltuu","Caaltuu-GEN"), ("afaanii","afaan-GEN"), ("namichaa","namicha-DAT"), ("intalaaf","intala-DAT"), ("sareef","saree-DAT"), ("baruuf","baruu-DAT"), ("bishaaniif","bishaan-DAT"), ("sareedhaa","saree-DAT"), ("sareedhaaf","saree-DAT"), ("Caaltuutti","Caaltuu-DAT"), ("harkaan", "harka-INST"), ("halkaniin", "halkan-INST"), ("Oromotiin", "Oromo-INST"), ("yeroodhaan", "yeroo-INST"), ("bawuudhaan", "bawuu-INST"), ("Arsiitti", "Arsii-LOC"), ("harkatti", "harka-LOC"), ("guyyaatti", "guyyaa-LOC"), ("jalatti", "jala-LOC"), ("biyyaa", "biyya-ABL"), ("Finfinneedhaa", "Finfinnee-ABL"), ("Hararii", "Harar-ABL"), ("bunaatii", "bunaa-ABL"), ("manoota", "mana-PL"), ("hiriyoota", "hiriyaa-PL"), ("barsiisota", "barsiisaa-PL"), ("barsiisoota", "barsiisaa-PL"), ("waggaawwan", "waggaa-PL"), ("laggeen", "laga-PL"), ("ilmaan", "ilma-PL"), ("namicha", "nama-M.DEF"), ("muzicha", "muzii-M.DEF"), ("durbittii", "durba-F.DEF"), ("ilkaan", "ilka-PL"), ("waantoota", "waanta-PL"), ("guyyawwan", "guyyaa-PL"), ("gaarreen", "gaara-PL"), ("laggeen", "laga-PL"), ("mukkeen", "muka-PL"), ("waggottii", "waggaa-PL"), ("kitaabolii", "kitaaba-PL"), ("yunivarsitichaan", "yunivarsiti-DEF-GEN"), ("adeema", "m"), ("adeemta", "a"), ("adeemti", "a"), ("adeemna", "a"), ("Finfinne", "Addis Ababa"), ("Biraazil", "brazil")]

#for pair in pairs:
#	parse = str(PARSER.parse(pair[0]))
#	if pair[1] not in parse:
#		print(pair[0], parse)

#print PARSER.parse("hidiin")
#print PARSER.parse("haattii")
#print PARSER.parse("duresii")
#print PARSER.parse("namnii")

if __name__ == '__main__':
    # just for testing.  to use this file, import it as a library and call parse() 
    with open("text-output.txt", "w", encoding="utf-8") as fout:
	testprint = lambda x: print(json.dumps(parse(x, "gloss"), indent=2, ensure_ascii=False), file=fout)
	#testprint("laggeen")	
	for pair in pairs:
		#print(pair)
		print([best_parse(pair[0], "gloss"), best_parse(pair[0], "lemma"), best_parse(pair[0], "natural")])
		print(parse(pair[0], "gloss"))
        	#print("HERE!!!",parse(pair[0], "gloss"))
		testprint(pair[0])
	 
