#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Na-Rae Han, naraehan@pitt.edu, August 2017 
# version 5.4
# incorporates noun internal plural morphology
# output in tir-Ethi-pp (still using tir-Ethi internally)
# new and updated user-facing functions:
#    fullparse, best_fullparse, parse, best_parse

from __future__ import print_function
from __future__ import unicode_literals

from io import open
import sys, json, glob, os, math
from copy import deepcopy
from morpar import *
import nltk
from nltk.probability import *
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
import re

def log_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

######################################
#
# EPITRAN, DICTIONARY FILE PATHS
#
######################################

#dict_path = "/home/data/LoReHLT17/internal/Morph/Tir/v5/" # on miami
#dict_path = "/usr2/data/shared/LoReHLT17/internal/Morph/Tir/v5/"  # on lor
dict_path = "/Users/aditichaudhary/Documents/CMU/Lorelei/LORELEI_NER/utils/segnerfts/res/tir/"  # on lor
#dict_path = "D:\\Projects\\LORELEI Surprise Language\\morphology\\v5\\"

# output IPA format: True (tir-Ethi-pp), False (tir-Ethi which is used internally)
# ***** Make sure that pre-parsed files are also in the right format!! *****
out_tir_pp = True  

import epitran
# original mapping: only token-final ɨ removed
# single-char geez script 'tɨ' returned as 't' in an earlier version. 
g2p = epitran.Epitran("tir-Ethi").transliterate 

# precision-phonemic: some ɨ removed internally, rendeing CC.  
# in a later version, single-char-final 'ɨ' not chopped, so 'tɨ' returned
g2pp = epitran.Epitran("tir-Ethi-pp").transliterate 
                
from epitran.tir2pp import Tir2PP
t2p = Tir2PP()
#p2pp = t2p.apply     # converts original p output to pp output.
                      # Nope, is a problem for multi-word input:
                      # "tɨɡɨrɨɲa tɨɡɨrɨɲa"  ->  "tɨɡrɨɲa tɡɨrɲa"
# tokenize, convert each token, and stitch them back 
p2pp = lambda txt: ' '.join([t2p.apply(x) for x in txt.split()])
    
######################################
#
# MORPAR DECLARATIONS
#
######################################

Text = Concatenated('text')
Breakdown = Hyphenated('breakdown')
Aff = Text / Breakdown
Def = Concatenated("definition")
Nat = Concatenated("natural")
Cost = Concatenated("cost")
Gloss = Hyphenated("gloss")
Lemma = Concatenated("lemma")
Lem = Text / Breakdown / Gloss / Lemma / Def / Nat

DEFAULTS.Text = Text
DEFAULTS.Lem = Lem
    
######################################
#
# DICTIONARY BUILDING
#
# This section builds three dictionary
# objects from various dictionary files. 
#
######################################

setSfile = dict_path+"setS_wordlist.txt"
setSWords = open(setSfile, "r", encoding="utf8").read().split()
        
def get_freq_dist():
    freq = FreqDist()
    freq.update(brown.words())
    freq.update(setSWords)
    return freq

l1_to_l2 = defaultdict(list)    # dictionary at top level, so it can be referenced
ncroot_to_l2 = defaultdict(list)
    
def make_dictionary(dict_filename_list, outdict):
    for dict_filename in dict_filename_list:
        try:
            with open(dict_filename, "r", encoding="utf-8") as fin:
                for line in fin.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        log_error("WARNING: Insufficient parts for line: %s (Safe to ignore if empty.)" % line)
                        continue
                    definition = parts[0]
                    word = parts[1]
                    ipa = g2p(word)     # tir-Ethi used internally
                    outdict[ipa].append(definition)
        except IOError:
            log_error(dict_filename, "was not found. Please let Na-Rae know.")
            continue

def make_root_dictionary(dict_filename_list, outdict):
    for dict_filename in dict_filename_list:
        try:
            with open(dict_filename, "r", encoding="utf-8") as fin:
                for line in fin.readlines():
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        log_error("WARNING: Insufficient parts for line: %s (Safe to ignore if empty.)" % line)
                        continue
                    definition = parts[0]
                    root = parts[1]
                    word = parts[2]
                    ipa = g2p(word)     # tir-Ethi used internally
                    outdict[root].append(  (ipa, definition)  ) # root dictionary.
                                    # key is root, value is (word, en_definition) list. 
        except IOError:
            log_error(dict_filename, "was not found. Please let Na-Rae know.")
            continue

def make_trivial_parse(br, le, gl, na, df, co):
    "Build and return a trivial parse from provided text values."
    prs = HashableDict()
    prs['breakdown'] = Breakdown.typ(br)
    prs['lemma'] = Lemma.typ(le)
    prs['gloss'] = Gloss.typ(gl)
    prs['natural'] = Nat.typ(na)
    prs['definition'] = Def.typ(df)
    prs['cost'] = Def.typ(co)
    return prs

def process_preparsed_dict(dict_filename_list):
    # Wow. What a mess. Might have to re-format the PREPARSED text files. 
    preparsed = defaultdict(list)

    for dict_filename in dict_filename_list :
        try:
            with open(dict_filename, "r", encoding="utf-8") as fin:
                for line in fin.readlines():
                    if line.startswith("#"): continue  # ignore comment lines
                    parts = line.split("\t")
                    # TIR, IPA, Gloss, Natural, Lemma, Breakdown, Comments
                    if len(parts) < 7:
                        log_error("WARNING: Insufficient parts for line: %s (Safe to ignore if empty.)" % line)
                        continue
                    tir, ipa, gloss, natural, lemma, breakdown, comment = tuple(parts)
                    ipa = g2p(tir)               # do not use IPA on file! 
                    ipa_out = g2pp(tir) if out_tir_pp else ipa

                    if breakdown == "": breakdown = ipa_out                    
                    if gloss == "": gloss = ipa_out
                    defin = "" # stays empty for HORN MORPHO guessed root 
                    cost = ""  # shall not stay empty 
                    if lemma == "":  # lemma is empty; hand-entered preparse
                        lemma = ipa_out
                        defin = natural  # "you are (m)", etc.
                        cost = "XXX"
                    else:     # HORN MORPHO: lemma is present as TIR word
                        gloss1st = gloss[:gloss.index('-')]   # beginning part of gloss
                        if gloss1st != g2pp(lemma): # gloss starts with eng word --> found in dict!
                                 # note that the file ipa notation is in tir-Ethi-pp, so use g2pp. 
                            defin = gloss1st
                            cost = "XXXXXX"
                        else: # lemma did not hit in the dict. empty defin, higher cost. 
                            cost = "XXXXXXXXXXXXXXXXXXXXX"
                        lemma = g2pp(lemma) if out_tir_pp else g2p(lemma)       # tir-Epi-pp conversion          

                    #preparsed[ipa] = [{'breakdown':breakdown, 'lemma':lemma, 'gloss':gloss, 'natural':natural}]
                    preparsed[ipa] = [ make_trivial_parse(breakdown, lemma, gloss, natural, defin, cost) ]
        except IOError:
            log_error(dict_filename, "was not found. Please let Na-Rae know.")
            continue
    return preparsed
            
# Below are dictionary files. First two fields are absolutely necessary: eng_definition, tir_word.
# 3rd column is IPA, in tir-Ethi. Not utilized by this module; it's for human readability only. 
dict_list = [dict_path+"IL5_dictionary_1.txt", dict_path+"IL5_dictionary_2.txt",
             dict_path+"IL5_dictionary_3_SUPPL.txt", dict_path+"IL5_dictionary_4_eng-tig_dict.txt",
             dict_path+"IL5_dictionary_5_tir_panlex.tsv", dict_path+"IL5_dictionary_6_tir_knight_lexicon.tsv",
             dict_path+"IL5_dictionary_7_DLIFLC.txt", 
             dict_path+"tir_gaz.txt", 
             dict_path+"lexicon_supplement.txt" ]  
make_dictionary(dict_list, l1_to_l2)

# noun consonant roots, for internal plural. No vowels, lists CCC only. 
make_root_dictionary([dict_path+"noun-consonant-roots.txt"], ncroot_to_l2)

# These files list fully parsed entries. 
dict_preparsed = [dict_path+"IL5_PREPARSED_hornmorpho.tsv", dict_path+"IL5_PREPARSED.tsv"] # order! 
preparsed = process_preparsed_dict(dict_preparsed)

######################################
#
# LOOKUP PARSER
#
# This section defines a custom parser
# that looks up a word in a dictionary
# object. 
#
######################################

class Lookup(Parser):

    def __init__(self, di, channel=None, output_channel=None, process_root=False):
        self.dictionary = di
        self.channel = channel
        self.output_channel = output_channel
        self.freqDist = get_freq_dist()
        self.engWords = self.freqDist.N()
        self.procroot = process_root

        # channel: Text/Breakdown/Lemma, output_channel: Gloss/Nat        
    
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
            ipaout = text
            if out_tir_pp: ipaout = p2pp(ipaout)    # conversion to 'tir-Ethi-pp'
            output[channel.name] = channel.typ(ipaout)
            
        if text in self.dictionary:
            for defin in self.dictionary[text]:
                definition = defin
                fullroot = ''        # will be unused unless processing root dictionary 
                if self.procroot:    # processing a root dictionary     
                    fullroot, definition  = defin[0], defin[1]  # (fullroot, engword) is value
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
                output2[Cost.name] = Cost.typ("X" * int(cost))
                if self.procroot:                              # if processing a root dict
                    output2[Lemma.name] = Lemma.typ(fullroot)  # provide fullroot as lemma, not CCC 
                    #output2[Breakdown.name] = Breakdown.typ(fullroot)  # nope, no need
                    #output2[Gloss.name] = Gloss.typ(fullroot)          # likewise
                results.add((output2,remnant))
        else:
            #print("didn't find it: %s" % text)
            output2 = deepcopy(output)
            for channel in self.output_channel:
                ipaout = text.replace(" ","")
                if out_tir_pp: ipaout = p2pp(ipaout)  # conversion to 'tir-Ethi-pp'
                output2[channel.name] = channel.typ(ipaout)
            cost = "X" * (50 + len(text))
            output2[Cost.name] = Cost.typ(cost)
            output2[Def.name] = channel.typ("")    # word not found; definition empty
            results.add((output2,remnant))
            
        return results
    
###############################
#
# MORPHOLOGICAL GRAMMAR
#
###############################

"""1 2 3   PL SG   MASC FEM
PERF IMPERF PRES
ITER GERNDV IMP REL PASS RECIP TRANS PREP
Q NEG"""

C = "b|d|d͡ʒ|f|h|hʷ|j|k|kʷ|l|m|n|p|pʼ|q|qʰ|qʰʷ|qʷ|r|s|t|tʼ|t͡sʼ|t͡ʃ|t͡ʃʼ|v|w|x|xʷ|z|ħ|ɡ|ɡʷ|ɲ|ʃ|ʒ|ʔ|ʕ"
V = "a|e|i|ɨ|o|u|ə"
Vtrue = "a|e|i|o|u|ə"

V_INITIAL = ( Before('ə') | Before('ɨ') | Before('a') | Before('i') | 
            Before('o') | Before('u') | Before('e') )

V_FINAL = ( After('ə') | After('ɨ') | After('a') | After('i') | 
            After('o') | After('u') | After('e') )

TRUNC_PLU_OT = ( Truncate("aj", Tex) | Truncate("a", Tex))
TRUNC_PLU_WITI = Truncate("t", Tex)

# /ɨ/ is an epenthetic vowel; Epitran "tir-Ethi" drops it in word-final positions only. 
# Below works, but it allows padding of i in whole-root guess. Cost assigned to "ghost i"
# TRUNC_FINALV = ( Truncate("ɨ", Tex) + Cost("XXXXXXXX")   # Cost must be on this. CHECKED. 
#                | NULL )
# Best solution (so far) below. No need for cost assigning. tɨ > t by cost of 1 X!  And no iɨ in sight. 
I_FINAL = After('ɨ')
SUFCONS = ( After('t') | After('m') | After('n') | After('j') )
       # these are the only suffix-final consonants I am using, which will be a stray V word-finally 
TRUNC_FINAL_I = ( Truncate("ɨ", Tex/Lem) + SUFCONS   # ordering crucial
                | ~I_FINAL )
########### Decided against the approach of mandating V-final suffixes (i.e., not otɨ but ot)
########### This rule is therefore unused.

INSERT_I = ( ~V_FINAL + Tex("ɨ")   # if stem ends in C, insert ɨ (before a C-initial suffix)
             | V_FINAL) 

REL = ( Aff('zɨ') + Gloss('REL') + Nat("which (.*)")
        | NULL )

CONJ_PREF = ( Aff('ki') + Gloss('CONJ') + Nat("and (.*)")
        | Aff('mɨ') + Gloss('CONJ') + Nat("and (.*)")
        | NULL )
#CONJ = (  # 'ʔaləjə' -> 'mələləj' Conjunction
# ('ʔɨtəxətəlom', 'təxətələ', 'follow-SBJ.3.SG.MASC-OBJ.3.PL.MASC-MOOD.PERF.PASS.REL', 'which did was follow')

NEG = ( Tex('ʔaj(.+)ɨn') + Gloss('NEG') + Nat("not (.*)") 
        | Tex('ʔaj(.+)n') + Gloss('NEG') + Nat("not (.*)") 
        | Tex('ʔajɨ(.+)n') + Gloss('NEG') + Nat("not (.*)") 
        | Tex('ʔajɨ(.+)ɨn') + Gloss('NEG') + Nat("not (.*)") 
        | Tex('ʔajɨtɨ(.+)ɨn') + Gloss('NEG') + Nat("don't (.*)")  # negative imperative
        | NULL )
    
TENSE = ( INSERT_I + Aff('kɨ') + Gloss('FUT') + Nat ("will (.*)")
    | INSERT_I + Aff('tə') + Gloss('PERF') + Nat ("(.*)-ed")
    | INSERT_I + Aff('jɨ') + Gloss('PERF') + Nat ("(.*)-ed")
    | NULL )

NUMBER = (
    ~V_FINAL + Aff('at') + Gloss('PL') + Nat("multiple (.*)") + Cost("XXXXXXXXXXXXXX") # after C (ዓራት > ዓራታት) (ʕarat - at) 'beds'
    | V_FINAL + Aff('tat') + Gloss('PL') + Nat("multiple (.*)")   # after V (እምባ > እምባታት) (ʔɨmɨba - tat) 'mountains'
    | TRUNC_PLU_OT + Aff('ot') + Gloss('PL') + Nat("multiple (.*)")   # following deletion of -a or -aj
                                                                      # (ሓረስታይ > ሓረስቶት) (ħarəsɨtaj > ħarəsɨtot )
    | (V_FINAL | TRUNC_PLU_WITI) + Aff('wɨti') + Gloss('PL') + Nat("multiple (.*)")
                                             # after V (ገዛ > ገዛውቲ) (ɡəza-wɨti) 'houses'
                                             # or t which gets deleted (ዓራት > ዓራውቲ) (ʕarat > ʕarawɨti) 'beds'
    | ~V_FINAL + Aff('ɨti') + Gloss('PL') + Nat("multiple (.*)")   # after C
    
    | NULL )

# Below from https://en.wikipedia.org/wiki/Tigrinya_grammar
# Need to take care of POL (polite) forms. 
POSS = (
      ~V_FINAL + Aff('əj') + Gloss('1SG.POSS') + Nat("my (.*)")    # after C
    |  V_FINAL + Aff('j') + Gloss('1SG.POSS') + Nat("my (.*)") + Cost("XXXX") # after V
    | INSERT_I + Aff('ka') + Gloss('2SG.MASC.POSS') + Nat("your (.*)") + Cost("XXXX")
    | INSERT_I + Aff('ki') + Gloss('2SG.FEM.POSS') + Nat("your (.*)")
    | ~V_FINAL + Aff('u') + Gloss('3SG.MASC.POSS') + Nat("his (.*)") + Cost("XXXX") # after C
    |  V_FINAL + Aff('ʔu') + Gloss('3SG.MASC.POSS') + Nat("his (.*)")   # glottal after V
    | ~V_FINAL + Aff('a') + Gloss('3SG.FEM.POSS') + Nat("her (.*)") + Cost("XXXXXXX")  # after C
    |  V_FINAL + Aff('?a') + Gloss('3SG.FEM.POSS') + Nat("her (.*)")   # glottal after V
    | INSERT_I + Aff('na') + Gloss('1PL.POSS') + Nat("our (.*)") + Cost("XXXXX") 
    | INSERT_I + Aff('kum') + Gloss('2PL.MASC.POSS') + Nat("your (.*)")
    | INSERT_I + Aff('kən') + Gloss('2PL.FEM.POSS') + Nat("your (.*)")
    | ~V_FINAL + Aff('om') + Gloss('3PL.MASC.POSS') + Nat("their (.*)")  # after C
    |  V_FINAL + Aff('?om') + Gloss('3PL.MASC.POSS') + Nat("their (.*)")  # glottal after V
    | Aff('ən') + Gloss('3PL.FEM.POSS') + Nat("their (.*)")
    | INSERT_I + Aff('?en') + Gloss('3PL.FEM.POSS') + Nat("their (.*)")
    | NULL  )

PRONCLITIC_OBLIQ = (
      INSERT_I + Aff('ləj') + Gloss('1SG.OBL') + Nat("(.*) to me")   # well represented
    | INSERT_I + Aff('lɨka') + Gloss('2SG.MASC.OBL') + Nat("(.*) to you") # lots of fusion, not very common
    | INSERT_I + Aff('lɨki') + Gloss('2SG.FEM.OBL') + Nat("(.*) to you") # only 3! 
    | INSERT_I + Aff('lu') + Gloss('3SG.MASC.OBL') + Nat("(.*) to him") # lots of fusion, some mulu/kumulu
    | INSERT_I + Aff('la') + Gloss('3SG.FEM.OBL') + Nat("(.*) to her") + Cost("XX") # well represented, relatively clean break. 
    | INSERT_I + Aff('lɨna') + Gloss('1PL.OBL') + Nat("(.*) to us") # well represented
    | INSERT_I + Aff('lɨkum') + Gloss('2PL.MASC.OBL') + Nat("(.*) to you") # 20? looks ok
    | INSERT_I + Aff('lɨn') + Gloss('2PL.FEM.OBL') + Nat("(.*) to you") # almost always involves ə -> ɨ on stem, l could be part of stem
    | INSERT_I + Aff('lom') + Gloss('3PL.MASC.OBL') + Nat("(.*) to them") # almost always lə -> lom on stem. 
    | INSERT_I + Aff('lən') + Gloss('3PL.FEM.OBL') + Nat("(.*) to them") + Cost("XXXX") # relatively clean break when occurs.
                                                      # but lots of false positives with lə+n
    | NULL )

# Additional forms need entering. 
PRONCLITIC_OBJ = (
      INSERT_I + Aff('ni') + Gloss('1SG.OBJ') + Nat("(.*) me") 
    | INSERT_I + Aff('ka') + Gloss('2SG.MASC.OBJ') + Nat("(.*) you") + Cost("XXXX")
    | INSERT_I + Aff('ki') + Gloss('2SG.FEM.OBJ') + Nat("(.*) you")  # very few of them
    | INSERT_I + Aff('jo') + Gloss('3SG.MASC.OBJ') + Nat("(.*) him")  # najo also frequent
    | INSERT_I + Aff('ʔo') + Gloss('3SG.MASC.OBJ') + Nat("(.*) him") 
    | INSERT_I + Aff('wo') + Gloss('3SG.MASC.OBJ') + Nat("(.*) him")  # lots of kɨwo
    | INSERT_I + Aff('ja') + Gloss('3SG.FEM.OBJ') + Nat("(.*) her")
    | INSERT_I + Aff('wa') + Gloss('3SG.FEM.OBJ') + Nat("(.*) her")
    | INSERT_I + Aff('ʔa') + Gloss('3SG.FEM.OBJ') + Nat("(.*) her") + Cost("XXXX") # Lots of fusion
    | INSERT_I + Aff('na') + Gloss('1PL.OBJ') + Nat("(.*) us") + Cost("XXXX")  # lɨna also frequent
    | INSERT_I + Aff('kum') + Gloss('2PL.MASC.OBJ') + Nat("(.*) you")   # lɨkum also common
    | INSERT_I + Aff('kɨn') + Gloss('2PL.FEM.OBJ') + Nat("(.*) you")   # **0** tokens! Huh. 
    | INSERT_I + Aff('jom') + Gloss('3PL.MASC.OBJ') + Nat("(.*) them")   # Oh dear. jə/wə -> jom. MESSY. 
    | INSERT_I + Aff('ʔom') + Gloss('3PL.MASC.OBJ') + Nat("(.*) them")   # Likewise. ʔə -> ʔom. No clean boundary. 
    | INSERT_I + Aff('jən') + Gloss('3PL.FEM.OBJ') + Nat("(.*) them") + Cost("XXXX")
                                    # infrequent. Most string matches are false positives: j+ən
    | NULL )     
    
PREP = ( Aff('bɨ') + Gloss('PREP') + Nat("with (.*)") 
        | Aff('nɨ') + Gloss('PREP') + Nat("for (.*)")
        | Aff('bɨzəjɨ') + Gloss('PREP') + Nat("without (.*)") 
        | Aff('mɨsɨ') + Gloss('PREP') + Nat("with (.*)") 
        | Aff('kabɨ') + Gloss('PREP') + Nat("from (.*)") 
        | Aff('kəmɨ') + Gloss('PREP') + Nat("like (.*)") 
        | Aff('dɨħɨri') + Gloss('PREP') + Nat("after (.*)") 
        | Aff('qɨdɨmi') + Gloss('PREP') + Nat("before (.*)") 
        | Aff('bɨzaʕɨba') + Gloss('PREP') + Nat("about (.*)") 
        | NULL )
"""
From Grammatical Sketch, not put in: 
ናይ naj nay    'of'
ምእንቲ mɨʔɨnɨti mǝ’ǝnti , ስለ sɨlə sǝlä    'for, because of, on the part of'
ክሳዕ kɨsaʕ kǝsa‘ , ክሳብ kɨsab kǝsab , ስጋዕ sɨɡaʕ sǝga‘    'until'
"""

# ʔɨ ʔɨtɨ ʔɨnɨ ʔɨtə
# These accompany internal vowel change and therefore did not help recall at all, except for ?i
# But should still be important to chop them off, for later verbal template matching. 
VDERIV_PREF = ( INSERT_I + Aff('ʔɨtɨ') + Gloss('REL') + Nat("which (.*)") 
             | INSERT_I + Aff('ʔɨnɨ') + Gloss('1PL.REL') + Nat("which (.*)") 
             | INSERT_I + Aff('ʔɨtə') + Gloss('PASS') + Nat("which was (.*)-ed") 
             | INSERT_I + Aff('ʔɨ') + Gloss('PRES') + Nat("(.*)")  # Not clear if indep. pref, may overgenerate -> HELPS! 
                                 # ʔɨfətɨwa እየ ʔɨjə ’ǝfätwa ’ǝyyä 'I like her'.  
             | NULL )

CASE_SUF = ( INSERT_I + Aff('n') + Gloss('ACC') + Nat("(.*)-OBJ") + Cost("XXXXXXXXXXXX") # bigger cost than 'and'
           | NULL )

CONJ_SUF = ( INSERT_I + Aff('wɨn') + Gloss('CONJ') + Nat("and also (.*)") # Clausal; attaches to verb. Final ɨ omitted for now. Post-C allomorph?
           | INSERT_I + Aff('n') + Gloss('CONJ') + Nat("(.*) and") + Cost("XXXXXXXXXX") # NP conjunction! lower cost than 'ACC' 
           | NULL )

ADJECTIVAL  = ( ~V_FINAL + Aff('awi') + Gloss('ADJ') + Nat("(.*)-ian") # in ሰኔጋላዊ səneɡalawi 'A Senegalese' Not sure if adj
    | After('i') + Aff('jawi') + Gloss('ADJ') + Nat("(.*)-ian")            
    | After('a') + Aff('wi') + Gloss('ADJ') + Nat("(.*)-ian")         
    | NULL )   

# ʔakadəmijawi  if ending in i, insert j then awi
# if ending in a, just wi
# mɨmɨħɨdar + awi "administration" consonant end --> awi
# haven't seen any other vowel + wi for adj. əwi seems to indicate noun...?
# ውያን   wɨjan    ዊያን  wijan

NOMINAL = (
      Aff('wɨjan') + Gloss('NOM.MASC') + Nat("(.*) person")    # in ኤርትራውያን ʔerɨtɨrawɨjan Eritrean national
    | Aff('wijan') + Gloss('NOM') + Nat("(.*) person")      # haven't seen myself, but jiatengx requested chopping
    | Aff('wit') + Gloss('NOM.FEM') + Nat("(.*) person")   # ኤርትራዊት ʔerɨtɨrawit female Eritrean national 
    | NULL) 

Mutate = lambda x, y :  Tex(y) + Truncate(x, Tex) 
MERGE = ( Mutate('kɨʔa', 'kə') | Mutate('mɨʔa', 'mə') | Mutate('zɨʔa', 'zə')
        | Mutate('bɨʔa', 'bə') | Mutate('nɨʔa', 'nə')
        | NULL) 

ROOT   = Lookup(l1_to_l2, Text/Breakdown/Lemma/Gloss, Nat/Def, False)
#ROOT = Guess(Lem)   # SWITCH FOR GRAMMAR BUILDING!! 

#======================================================= INTERNAL PLURAL STUFF
#                                                        From Grammar Sketch: 
# ’aCCaC    =>      ʔ a C ɨ C a C ə      or  ʔ a C ɨ C a C      
# no discernible pattern on singular side
"""
ፈረስ fərəs  'horse', ኣፍራሰ ʔafɨrasə  'horses'
እዝኒ ʔɨzɨni  'ear', ኣእዛን ʔaʔɨzan  'ears'
"""
#plpat1 = 'ʔ a C ɨ C a C ə'.replace('C', '(.+)').replace(' ','')
#plpat1a = 'ʔ a C ɨ C a C'.replace('C', '(.+)').replace(' ','')
plpat1 = 'ʔa(%s)ɨ(%s)a(%s)ə' % (C, C, C)
plpat1a = 'ʔa(%s)ɨ(%s)a(%s)' % (C, C, C)


# ’aCaCǝC   =>     ʔ a C a C ɨ C
# maybe the 2nd vowel has to be ɨ on singular side?
"""
ንህቢ nɨhɨbi  'bee', ኣናህብ ʔanahɨb  'bees'
በግዕ bəɡɨʕ  'sheep' (s.), ኣባግዕ ʔabaɡɨʕ  'sheep' (p.)
"""
plpat2 = 'ʔa(%s)a(%s)ɨ(%s)' % (C, C, C)

# CäCaCu    =>     C ə C a C u    or  C ɨ C a C u 
# no discernible pattern on singular side
"""
ደርሆ dərɨho  'chicken', ደራሁ dərahu  'chickens'
ጕሒላ ɡʷɨħila  'thief', ጕሓሉ ɡʷɨħalu  'thieves'
"""
plpat3 = '(%s)ə(%s)a(%s)u' % (C, C, C)
plpat3a = '(%s)ɨ(%s)a(%s)u' % (C, C, C)

# C{ä,a}CaCǝC   => C ə C a C ɨ C     or   C a C a C ɨ C
# SG: 2nd vowel ɨ, no 3rd (final) vowel
"""
መንበር mənɨbər  'chair', መናብር mənabɨr  'chairs'
ሓርማዝ ħarɨmaz  'elephant', ሓራምዝ ħaramɨz  'elephants'
"""
plpat4 = '(%s)ə(%s)a(%s)ɨ(%s)' % (C, C, C, C)
plpat4a = '(%s)a(%s)a(%s)ɨ(%s)' % (C, C, C, C)

# ...äCti   =>  ...ə C ɨti
# for the plural of agent and instrument nouns derived from verbs
# SG: ends in (a|ə)Ci
"""
ቀላቢ qəlabi  'feeder', ቀለብቲ qələbɨti  'feeders'
ኣገልጋሊ ʔaɡəlɨɡali  'server', ኣገልገልቲ ʔaɡəlɨɡəlɨti  'servers'
መኽደኒ məxɨdəni  'cover', መኽደንቲ məxɨdənɨti  'covers'
"""
plpat5_no_fixed_length = True  # leaving for later implementation

# CǝCawǝCti     => C ɨ C awɨ C ɨti
# SG: CɨCan form? At least both examples are...
"""
ክዳን kɨdan  'clothing', ክዳውንቲ kɨdawɨnɨti kɨda-wɨ-n-ɨti  'articles of clothing'
ሕጻን ħɨt͡sʼan  'infant', ሕጻውንቲ ħɨt͡sʼawɨnɨti  ħɨt͡sʼa-wɨ-n-ɨti  'infants'
"""
plpat6 = '(%s)ɨ(%s)awɨ(%s)ɨti' % (C, C, C)

# CäCaCǝCti     =>  C ə C a C ɨ C ɨti
# SG: ends in C
"""
መጽሓፍ mət͡sʼɨħaf  'book', መጻሕፍቲ mət͡sʼaħɨfɨti  'books'
ኮኸብ koxəb  'star', ከዋኽብቲ kəwaxɨbɨti  'stars'
"""
plpat7 = '(%s)ə(%s)a(%s)ɨ(%s)ɨti' % (C, C, C, C)

# ...C*aC*ǝC...   ... C* ə C  => ...C* a C* ɨ C i      or ...C* a C* ɨ C   
# reduplicating a single root consonant with /-a-/ (represented by "C*aC*")
# SG: ends in ə C ə C 
"""
ወረቐት wərəqʰət  'paper', ወረቓቕቲ wərəqʰaqʰɨti  'papers'
ተመን təmən  'snake', ተማምን təmamɨn  'snakes'
"""
plpat8_no_fixed_length = True   # this one involves reduplication too. 
                                # leaving for later implementation.  

PLU_PATTERN_ALL = ( Tex(plpat1) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat1a) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat2) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat3) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat3a) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat4) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat4a) + Glo("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat6) + Gloss("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                  | Tex(plpat7) + Glo("PL") + Nat("Multiple-(.*)") + Cost("XXXXX")
                    )       ##### cost assigned, due to ambiguity noted below.

# 3043 total eligible (3-consonant) noun roots, 1845 consonant skeletons.
#     <--- ambiguity ratio of 1.65. Not as bad as I thought.    
NCROOT = Lookup(ncroot_to_l2, Text/Breakdown/Lemma, Gloss/Nat/Def, process_root=True)
N_INT_PLU = NCROOT << PLU_PATTERN_ALL

#========================================================= END Internal plural stuff

# REL is outer to NEG, which is taken care of by Mutate('zɨʔa', 'zə')
# zǝ- + ’ay- : zäy- , e.g., ዘይንደሊ zəjɨnɨdəli 'which we don't want'

CLITICS = (POSS | PRONCLITIC_OBLIQ | PRONCLITIC_OBJ)
MATRIX = (TENSE|PREP|VDERIV_PREF) >> (ROOT|N_INT_PLU) << NUMBER << CLITICS << NEG << (CASE_SUF|ADJECTIVAL|NOMINAL|CONJ_SUF)
# PARSER = MERGE >> (REL|CONJ_PREF) >> MATRIX << TRUNC_FINAL_I
PARSER = MERGE >> (REL|CONJ_PREF) >> MATRIX
#PARSER = (REL|CONJ) >> MATRIX 
# integrating two yields error for ዝክርን zɨkɨrɨn. Cannot do PARSER = REL >> FUT >> ROOT 


##############################
#
# FUNCTIONS
#
##############################

# A quick check on incoming token. Does it have an ASCII char somewhere?
# asciipat = re.compile(r'(a|b|c|d|e|f|g|h|i|j|k|l|l|m|o|p|q|r|s|t|u|v|w|x|y|z)', flags=re.IGNORECASE)
asciipat = re.compile(r'[A-Za-z]')
def is_ascii_token(tok):
    # clipped = unicode(tok[1:-1])      # clipping no longer necessary
    if asciipat.search(unicode(tok)):   # search in full token 
        return True
    else: return False

@lru_cache(maxsize=1000)
def fullparse(word, top=3, guess=True): 
    """
    Parses a Ge'ez word and returns an ordered list of morphological parses.
    
    Args: 
      word:    A word in Ge'ez script. May contain spaces and other symbols.
      (top=int): Number of top parses to return. Default is 3.
                 To get all parses, use 0. 
      (guess=bool): Whether to permit guessed stems. Default is True.
                    If set to False, all parses that did not terminate with
                    a dictionary-found stem will be discarded. If this leads
                    to an empty parse list, a list with a single default parse,
                    with no stemming applied, will be returned.
                
    Returns an ordered list of parses, in ascending order of cost, where
       each parse is a dictionary with the following keys for channels:
           "lemma", "gloss", "breakdown", "definition", "natural", "cost"
           
    lemma -- Stem after removing all affixes, e.g., ʔertɨrawjan --> ʔertɨra
    
    gloss -- Stem's meaning plus grammatical information provided by affixes
             "be-3SG.MASC"       3rd person singular masculine form of 'be'
             "təħħɨza-PL"        plural form of a guessed stem 'təħħɨza'
             
    breakdown -- Full form with morpheme boundaries indicated by '-'
                 "ʔertɨra-wɨjan", "mɨ-t͡sʼɨħɨf-ɨti"
    
    definition -- Lemma's meaning (in English) pulled from a dictionary.
                  Empty string "" if stem is guessed. 

    natural -- Natural-sounding English reading of the word. 

    cost -- Cost of the parse in string length. XXXXX indicates a very small
            cost (item could be directly off of a dictionary entry) which means
            high confidence; XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            indicates low confidence and a guessed stem with no dictionary hit. 
    """
    word = word.replace(r"\x94","").replace(r"\x93","")   # get rid of curly quotation marks. 

    # Check if input is in roman characters. 
    isascii = is_ascii_token(word)
    
    # Do some token clean up
    word = word.replace('”', '').replace('“', '')  # works on my end, not on miami.
    word = word.replace('\x94', '').replace('\x93', '') # doesn't work on miami. 
    word = word.replace('`', '')    

    ipa = g2p(word)
    ipa_out = ipa if not out_tir_pp else g2pp(word)
    parses = []

    # Some words will go through PARSER, some won't:
    if isascii:  # input is ASCII char. Word itself, empty definition/cost 
        parses = [ make_trivial_parse(word, word, word, word, "", "") ]

    elif ipa in l1_to_l2:    # whole word form found in dict file somewhere,
                             # including some 1-char prepositions. 
        parses = PARSER.parse(ipa)
        # reduce cost of whole-word parse if found in dictionary
        # these are essentially full-dictionary words.
        # some full-dictionary words come out ranked lower than analyzed out,
        # because Lookup routine has only local visibility during each parse step.
        # ስደተኛታት sɨdətəɲatat 'migrants' vs. 'multiple refugee' ("sɨdətəɲa-tat")
        for p in parses: 
            if g2pp(word) == p["breakdown"] and p["definition"] != "":
                original_cost = p["cost"]
                p["cost"] = original_cost[:-2]    # take off 2. should be just enough. 
    elif len(word) == 1:     # 1-char input not in dict: likely acronym, return IPA.
        pass                 # taken care of later
    elif ipa in preparsed:   # word is found in preparsed. just look it up.  
        parses = preparsed[ipa]
    else:                    # parse away!  
        parses = PARSER.parse(ipa)

    if not guess and not isascii:  # guess is turned off, and input is not ASCII. 
        # filter out guessed roots. Horn morpho's guess will also be thrown out.  
        parses  = [p for p in parses if p['definition'] != ""]

    if not parses:   # happens with single-char input or when all guesses are thrown away 
        # print("Warning: cannot parse %s (%s)" % (word, ipa))
        parses = [ make_trivial_parse(ipa_out, ipa_out, ipa_out, ipa_out, "", "") ]        
       
    parses.sort(key=lambda x:len(x["cost"]) if "cost" in x else 0)
    if top: return parses[:top]
    else: return parses

@lru_cache(maxsize=1000)
def best_fullparse(word):
    """Takes a word in Ge'ez script, returns the top-ranked morphological parse.
    Returned parse is a dictionary with the following keys for channels:
                "lemma", "gloss", "breakdown", "definition", "natural", "cost"
    To get a list of all parses or use more flexible options, use the
    fullparse() function instead.
    """
    return fullparse(word)[0]

@lru_cache(maxsize=1000)
def parse(word, channel="lemma"):
    """Takes a Ge'ez word and returns an ordered list of specified channel output.
    Parameters are: 
      word:    A word in Ge'ez script. 
      (channel=str): Morphological channel. Default is "lemma". Can be one of:
                "lemma", "gloss", "breakdown", "definition", "natural", "cost"
    To obtain full parses with all channels or use more flexible options, use the
    fullparse() function instead.
    """
    parses = fullparse(word)
    return [p[channel] for p in parses]

@lru_cache(maxsize=1000)
def best_parse(word, channel="lemma"):
    """Takes a Ge'ez word and returns top candidate (str) in specified channel output.
    Parameters are: 
      word:    A word in Ge'ez script. 
      (channel=str): Morphological channel. Default is "lemma". Can be one of:
                "lemma", "gloss", "breakdown", "definition", "natural", "cost"
    To obtain full parses with all channels or use more flexible options, use the
    fullparse() function instead.
    """    
    return parse(word, channel)[0]

def is_success(ps):
    """"parse list --> True/False
    Whether top parse is based on a dictionary entry i.e, not a guessed stem"""
    return ps[0]['definition'] != ""


##############################
#
# FOR TESTING CONVENIENCE
#
##############################

def jsonprint(someobj):
    "Quickly print a parse object, which is full of unicode chars. @Python2.7"
    print(json.dumps(someobj, ensure_ascii=False))

sample = [ 'ገዛውቲ',     # ɡəzawɨti "houses"
           'ሂብካዮ',     # hibɨkajo "find"
            'ዞናዊ',     # zonawi "regional"
           'ብዘይተፈልጡ',  # bɨzəjɨtəfəlɨtʼu  with a guessed stem.
           "ኣዲስ ኣበባ"     # Addis Ababa, multi-word input
           ]
# ---------------- Suggested tests:
# jsonprint(fullparse(sample[0]))
# jsonprint(best_fullparse(sample[0]))
# jsonprint(parse(sample[0], 'gloss'))
# jsonprint(best_parse(sample[0], 'gloss'))

# 'Ever since then , I have been moved from one prison to another until I was taken to
# detention camp in Tripoli .'
sample_text = "ካብቲ ንበልዖ መግብን ንሰትዮ ማይን ጀሚርካ ፡ ኣብ ማእሰርቲ ንሰብ ዘይግባእ ሕሱም ኣተሓሕዛ እዩ ዘለዎም ።"

# ---------------- Suggested tests:
# for w in sample_text.split(): jsonprint(parse(w, 'definition'))
# for w in sample_text.split(): jsonprint(parse(w, 'lemma'))
# for w in sample_text.split(): jsonprint(parse(w, 'cost'))
# for w in sample_text.split(): jsonprint(best_parse(w, 'natural'))
# for w in sample_text.split(): jsonprint(best_parse(w, 'gloss'))
# for w in sample_text.split(): jsonprint(best_fullparse(w))
# for w in sample_text.split(): jsonprint(fullparse(w))
# for w in sample_text.split(): jsonprint(fullparse(w, top=0))
# for w in sample_text.split(): jsonprint(fullparse(w, top=0, guess=False))





##############################
#
# MAIN, DEVELOPMENTAL ODDITIES
#
##############################

if __name__ == '__main__':
    # just for testing.  to use this file, import it as a library and call parse()

    jsonprint(fullparse(sample[0]))

    for w in sample_text.split(): jsonprint(parse(w, 'lemma'))

    for w in sample_text.split(): jsonprint(best_fullparse(w))




# 275 characters
allchars = """ሀ ሁ ሂ ሃ ሄ ህ ሆ ለ ሉ ሊ ላ ሌ ል ሎ ሐ ሑ ሒ ሓ ሔ ሕ ሖ መ ሙ ሚ
ማ ሜ ም ሞ ሠ ሡ ሢ ሣ ሤ ሥ ሦ ረ ሩ ሪ ራ ሬ ር ሮ ሰ ሱ ሲ ሳ ሴ ስ ሶ ሸ ሹ ሺ ሻ ሼ
ሽ ሾ ቀ ቁ ቂ ቃ ቄ ቅ ቆ ቈ ቊ ቋ ቌ ቍ ቐ ቑ ቒ ቓ ቔ ቕ ቖ ቘ ቚ ቛ ቜ ቝ በ ቡ ቢ ባ ቤ
ብ ቦ ቨ ቩ ቪ ቫ ቬ ቭ ቮ ተ ቱ ቲ ታ ቴ ት ቶ ቸ ቹ ቺ ቻ ቼ ች ቾ ኀ ኁ ኂ ኃ ኄ ኅ ኆ ኈ
ኊ ኋ ኌ ኍ ነ ኑ ኒ ና ኔ ን ኖ ኘ ኙ ኚ ኛ ኜ ኝ ኞ አ ኡ ኢ ኣ ኤ እ ኦ ከ ኩ ኪ ካ ኬ ክ ኮ ኰ
ኲ ኳ ኴ ኵ ኸ ኹ ኺ ኻ ኼ ኽ ኾ ዀ ዂ ዃ ዄ ዅ ወ ዉ ዊ ዋ ዌ ው ዎ ዐ ዑ ዒ ዓ ዔ ዕ
ዖ ዘ ዙ ዚ ዛ ዜ ዝ ዞ ዠ ዡ ዢ ዣ ዤ ዥ ዦ የ ዩ ዪ ያ ዬ ይ ዮ ደ ዱ ዲ ዳ ዴ ድ ዶ ጀ ጁ ጂ
ጃ ጄ ጅ ጆ ገ ጉ ጊ ጋ ጌ ግ ጎ ጐ ጒ ጓ ጔ ጕ ጠ ጡ ጢ ጣ ጤ ጥ ጦ ጨ ጩ ጪ ጫ ጬ ጭ
ጮ ጰ ጱ ጲ ጳ ጴ ጵ ጶ ጸ ጹ ጺ ጻ ጼ ጽ ጾ ፀ ፁ ፂ ፃ ፄ ፅ ፆ ፈ ፉ ፊ ፋ ፌ ፍ ፎ ፐ ፑ ፒ ፓ
ፔ ፕ ፖ"""

# g2p sometimes outputs ɨ for a single char:  t͡ʃʼɨ ɡʷɨ ɡɨ  
# Didn't realize that. Seems to be fixed in g2pp  (tir-Epi-pp). 
allchars_ipa = """hə hu hi ha he hɨ ho lə lu li la le lɨ lo ħə ħu ħi ħa ħe
ħɨ ħo mə mu mi ma me mɨ mo sə su si sa se sɨ so rə ru ri ra re rɨ ro sə su
si sa se sɨ so ʃə ʃu ʃi ʃa ʃe ʃɨ ʃo qə qu qi qa qe qɨ qo qʷə qʷi qʷa qʷe qʷɨ
qʰə qʰu qʰi qʰa qʰe qʰɨ qʰo qʰʷə qʰʷi qʰʷa qʰʷe qʰʷɨ bə bu bi ba be bɨ bo
və vu vi va ve vɨ vo tə tu ti ta te tɨ to t͡ʃə t͡ʃu t͡ʃi t͡ʃa t͡ʃe t͡ʃɨ t͡ʃo hə hu
hi ha he hɨ ho hʷə hʷi hʷa hʷe hʷɨ nə nu ni na ne nɨ no ɲə ɲu ɲi ɲa ɲe ɲɨ ɲo
ʔə ʔu ʔi ʔa ʔe ʔɨ ʔo kə ku ki ka ke kɨ ko kʷə kʷi kʷa kʷe kʷɨ xə xu xi xa xe
xɨ xo xʷə xʷi xʷa xʷe xʷɨ wə wu wi wa we wɨ wo ʕə ʕu ʕi ʕa ʕe ʕɨ ʕo zə zu zi
za ze zɨ zo ʒə ʒu ʒi ʒa ʒe ʒɨ ʒo jə ju ji ja je jɨ jo də du di da de dɨ do
d͡ʒə d͡ʒu d͡ʒi d͡ʒa d͡ʒe d͡ʒɨ d͡ʒo ɡə ɡu ɡi ɡa ɡe ɡɨ ɡo ɡʷə ɡʷi ɡʷa ɡʷe ɡʷɨ tʼə tʼu
tʼi tʼa tʼe tʼɨ tʼo t͡ʃʼə t͡ʃʼu t͡ʃʼi t͡ʃʼa t͡ʃʼe t͡ʃʼɨ t͡ʃʼo pʼə pʼu pʼi pʼa pʼe
pʼɨ pʼo t͡sʼə t͡sʼu t͡sʼi t͡sʼa t͡sʼe t͡sʼɨ t͡sʼo t͡sʼə t͡sʼu t͡sʼi t͡sʼa t͡sʼe t͡sʼɨ
t͡sʼo fə fu fi fa fe fɨ fo pə pu pi pa pe pɨ po"""


#38 consonants
cons = "b d d͡ʒ f h hʷ j k kʷ l m n p pʼ q qʰ qʰʷ qʷ r s t tʼ t͡sʼ t͡ʃ t͡ʃʼ v w x xʷ z ħ ɡ ɡʷ ɲ ʃ ʒ ʔ ʕ"
# b|d|d͡ʒ|f|h|hʷ|j|k|kʷ|l|m|n|p|pʼ|q|qʰ|qʰʷ|qʷ|r|s|t|tʼ|t͡sʼ|t͡ʃ|t͡ʃʼ|v|w|x|xʷ|z|ħ|ɡ|ɡʷ|ɲ|ʃ|ʒ|ʔ|ʕ

#7 vowels
vows = "a e i ɨ o u ə"
# a|e|i|ɨ|o|u|ə
