#!/usr/bin/env python
# -*- coding: utf-8 -*-


import regex as re


LONG_TOKEN_THRESH = 8


def ex_capitalized(ws):
    return [w[0].isupper() for w in ws]


def ex_all_uppercased(ws):
    return [all(x.isupper() for x in w) for w in ws]


def ex_mixed_case(ws):
    def mixed_case(w):
        noninit = [x.isupper() for x in w[1:]]
        return True in noninit and False in noninit
    return map(mixed_case, ws)


def ex_internal_period(ws):
    return [len(w) > 2 and '.' in w[1:-1] for w in ws]


def ex_non_letter(ws):
    return [bool(re.search(r'[^\p{Letter}\p{Mark}]', w)) for w in ws]


def ex_digits(ws):
    return [bool(re.search(r'[\p{Number}]', w)) for w in ws]


def ex_long_token(ws):
    return [len(w) > LONG_TOKEN_THRESH for w in ws]


def ex_contains_latin(ws):
    return [bool(re.search(r'\p{Latin}', w)) for w in ws]


def ex_contains_ethiopic(ws):
    return [bool(re.search(r'\p{Ethiopic}', w)) for w in ws]


ex_title = {
    'eng': lambda ws: [False] + [w in {
        'Mister',
        'Mr.',
        'Mr',
        'Misses',
        'Mrs.',
        'Mrs',
        'Miss',
        'Ms.',
        'Ms',
        'Doctor',
        'Dr.',
        'Dr',
        'Professor',
        'Prof.',
        'Prof',
        'Father',
        'Fr.',
        'Fr',
        'Reverend',
        'Rev.',
        'Rev',
        'Revd',
        'Pastor',
        'Bishop',
        'Bp.',
        'Bp',
        'President',
        'Pres.',
        'Representative',
        'Rep.',
        'Rep',
        'Congressman',
        'Congresswoman',
        'Congressperson',
        'Senator',
        'Sen.',
        'Sen',
        'Secretary',
        'Sec.',
        'Sec',
        'Lord',
        'Lady',
        'Justice',
        'Sheriff',
        'Principal',
    } for w in ws[:-1]],
    'amh': lambda ws: [False] + [w in {
        'አቶ',
        'ወይዘሮ',
        'ወይዘሪት',
        'ፕሮፌሰር',
        'ፕሬዚዳንት',
        'ፐሬዝዳንት',
        'ፕሬዝዳንት',
        'ኮለኔል',
        'ጄኔራል',
        'አቡነ',
        'ቀስ',
        'ሰላም',
        'ሼኽ',
        'ራስ',
        'ቢትወደድ',
        'ወ/ሮ',
        'ወ/ሪት',
        'ድ/ር',
        'ፕ/ር',
        'ፕ/ት',
        'ኮ/ል',
        'ጄ/ል',
        'ሼኽ',
        'ራስ',
        'ቢትወደድ',
        'አዛዥና',
        'ልዑል',
        'ሚኒስቴር',
        'ዕድሜው',
        'ወታደር',
        'ም/ል',
        'ጸሃፊ',
        'ረዳት',
        'ጸሐፊ',
        'አምባሳደር',
        'አስተዳዳሪ',
        'ሪፖርተራችን',
    } for w in ws[:-1]],
    'orm': lambda ws: [False] + [w.lower() in {
        'obbo',  # Mister
        'obboo',  # Mister
        'obo',  # Mister
        'abbaa',  # Father
        'aba',
        'ministeeraa',  # Minister
    } for w in ws[-1]],
    'tir': lambda ws: [False] + [w in {
        'ኣቶ',  # Mister_1
        'ጐይታይ',  # Mister_2
        'ሓላፊ',  # President_1
        'ሓለቓ',  # President_2
        'ወዘተ',  # President_3
        'ፕረሲደንት',  # President_4
        'ፕሬዝዳንት',  # President_5
        'ኣቦ',  # Father
    } for w in ws[:-1]],
}


ex_head_org = {
    'eng': lambda ws: [w in {
        'Ministry',
        'Department',
        'Agency',
        'Bureau',
        'Company',
        'Corporation',
        'Authority',
        'Organization',
        'Organisation',
        'Committee',
        'Bank',
    } for w in ws],
    'amh': lambda ws: [w in {
        'ሚኒስቴር',
        'ኤጀንሲ',
        'ኮሚሽን',
        'ድርጅት',
        'ባለሥልጣን',
        'ባንክ',
        'ቢሮ',
        'ኮሚቴ',
        'ኮርፖሬሽን',
        'ምንጮች',
    } for w in ws],
    'orm': lambda ws: [w.lower() in {
        'ministirii',  # Ministry
        'ministiri',
        'damiyyaa',  # Department
        'damiyya',
        'wakkiila',  # Agency
        'wakila',
        'dhaabbata',  # Organization
        'dhabata',
        'koree',  # Committee
        'kore',
        'baankii',  # Bank
        'banki',
        'waldaa',  # Society
        'walda',
        'waraanni',  # Front
        'warnani',
    } for w in ws],
    'tir': lambda ws: [w in {
        'ክፍሊ',  # Department_1
        'ጨንፈር',  # Department_2
        'ዋኒን',  # Agency_1
        'ተግባር',  # Agency_2
        'ስርሒት',  # Agency_3
        'ሰደቓ',  # Bureau
        'ኮርፖረሽን',  # Corporation
        'ውድብ',  # Organization_1
        'ኣወዳድባ',  # Organization_2
        'ኣመሰራርታ',  # Organization_3
        'ኮመት',  # Committee_1
        'ሽማግለ',  # Committee_2
        'ሰራዊት',  # Army
    } for w in ws],
}


ex_head_loc = {
    'eng': lambda ws: [w in {
        'Island',
        'Lake',
        'River',
        'Sea',
        'Ocean',
        'Mountain',
        'Mountains',
        'Valley',
        'Bay',
        'Mosque',
    } for w in ws],
    'amh': lambda ws: [w in {
        'ደሴት',
        'ሐይክ',
        'ወንዝ',
        'ባህር',
        'ወቅያኖስ',
        'ተራራ',
        'ሸለቆ',
        'ሰፈር',
        'ወሽመጥ',
        'መስጊድ',
        'ሀገር',
    } for w in ws],
    'orm': lambda ws: [w.lower() in {
        'odoola',  # Island
        'odola',
        'odoolota',  # Islands
        'odolota',
        'calalaqa',  # Lake_1
        'dabbal',  # Lake_2
        'dabal',
        'hara',  # Lake_3
        'laaqii',  # Lake_4
        'laqi',
        'lagaa',  # River
        'laga',
        'garba',  # Sea
        'maanya',  # Ocean
        'manya',
        'gooroo',  # Mountains
        'goro',
        'gaara',  # Mountain
        'sulula',  # Valley
        'bataskaana',  # Church
        'masqiida',  # Mosque
    } for w in ws],
    'tir': lambda ws: [w in {
        'ደሴት',  # Island_1
        'ግሉል',  # Island_2
        'ብሕቱው',  # Island_3
        'ቀላይ',  # Lake_1
        'ወይናይ',  # Lake_2
        'ፈለግ',  # River
        'ባሕሪ',  # Sea
        'ሰፊሕ',  # Ocean
        'ጎቦ',  # Mountain_1
        'እምባ',  # Mountain_2
        'ሩባ',  # Valley_1
        'ለሰ',  # Valley_2
        'ሕሉም',  # Valley_3
        'ስንጭሮ',  # Valley_4
        'በተኽስያን',  # Church
        'መስጊድ',  # Mosque
    } for w in ws],
}


ex_head_gpe = {
    'eng': lambda ws: [w in {
        'District',
        'Zone',
        'Region',
        'Province',
        'Division',
        'Republic',
        'Nation',
        'City',
        'Town',
        'Village',
        'State',
    } for w in ws],
    'amh': lambda ws: [w in {
        'ከተማ',
        'መንደር',
        'ቀበሌ',
        'ወረዳ',
        'ዞን',
        'ክልል',
        'አውራጃ',
        'መንግስት',
        'ክፍላት',
        'ጦር',
        'ዙሪያ',
        'ላይ',
    } for w in ws],
    'orm': lambda ws: [w.lower() in {
        'koonyaa',  # District_1
        'konya',
        'aanaa',  # District_2
        'ana',
        'goltaa',  # Zone_1
        'golta',
        'godina',  # Zone_2
        'naannoo',  # Region
        'nano',
        'jamuriyaa',  # Republic_1
        'jamuriya',
        'republika',  # Republic_2
        'magaalaa',  # City
        'magala',
        'magaalaan',
        'magalan',
        'daabbaa',  # Town
        'daba',
        'dira',  # Big Town
        'gandaa',  # Village
        'ganda',
        'mootummaa',
        'motuma',
    } for w in ws],
    'tir': lambda ws: [w in {
        'ወረዳ',  # District
        'ዞባ',  # Zone
        'ከተማ',  # City
        'ዞና',  # Region
        'መንግስቲ',  # State
        'ኣውራጃ',  # Prefecture/Province
        'ረፑብሊክ',  # Republic
        'ከተማ',  # City
        'ገጠር',  # Village_1
        'ቁሸት',  # Village_2
        'ዓዲ',  # Village_3
    } for w in ws],
}


ex_prep_from = {
    'eng': lambda ws: [w.lower() == 'from' for w in ws],
    'amh': lambda ws: [bool(re.match('ከ', w)) for w in ws],
    'orm': lambda ws: [w.lower() in {'irraa', 'ira'} for w in ws],
    'tir': lambda ws: [w in {'ካብ'} for w in ws],
}


ex_prep_in = {
    'eng': lambda ws: [w.lower() == 'in' for w in ws],
    'amh': lambda ws: [bool(re.match('በ', w)) for w in ws],
    'orm': lambda ws: [w.lower() in {'keessa', 'kesa'} for w in ws],
    'tir': lambda ws: [w in {'ኣብ'} for w in ws],
}


extractors = [
    lambda lang: ex_capitalized,
    lambda lang: ex_all_uppercased,
    lambda lang: ex_mixed_case,
    lambda lang: ex_internal_period,
    lambda lang: ex_non_letter,
    lambda lang: ex_digits,
    lambda lang: ex_long_token,
    lambda lang: ex_contains_latin,
    lambda lang: ex_contains_ethiopic,
    lambda lang: ex_title[lang],
    lambda lang: ex_head_org[lang],
    lambda lang: ex_head_loc[lang],
    lambda lang: ex_head_gpe[lang],
    lambda lang: ex_prep_from[lang],
    lambda lang: ex_prep_in[lang],
]


TYPE_START, TYPE_END = 0, 9
TOKEN_START, TOKEN_END = 9, 15


def extract(lang, seg):
    fts = zip(*[ex(lang)(seg) for ex in extractors])
    return [map(int, f) for f in fts]


def extract_type_level(lang, seg):
    fts = extract(lang, seg)
    return [v[TYPE_START:TYPE_END] for v in fts]


def extract_token_level(lang, seg):
    fts = extract(lang, seg)
    return [v[TOKEN_START:TOKEN_END] for v in fts]

def extractIndicatorFeatures(lang, seg):
    fts = extract(lang, seg)
    return fts
