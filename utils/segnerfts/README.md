# Documentation for `segnerfts.py`

The module `segnerfts` defines NER indicator feature extractors for the following languages:

| Language | ISO 639-3 |
|----------|-----------|
| Amharic  | amh       |
| English  | eng       |
| German   | deu       |
| Oromo    | orm       |
| Somali   | som       |
| Tigrinya | tir       |

## Dependencies

This code requires the `unicodecsv` package.

## Usage

The function `extract` takes as arguments the ISO 639-3 code and a list of tokens (ideally, a sentence) and returns a list consisting of a list of feature values for each token in the input.

```python
>>> import segnerfts
>>> segnerfts.extract('deu', u'Vereinigten Arabischen Republik'.split())
[[1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]]
```

The functions `extract_type_level` and `extract_token_level` take arguments of the same types but return only type-level and token-level features, respectively. The function `extract_gaz_features` features takes arguments of the same type and returns only the gazetteer features.

## Features

The type-level feature extractors are functions. The token-level features are dictionaries that take ISO 639-3 codes and return functions.

### Type-Level Features

* `ex_capitalized`: is the first character of the token upper-case?
* `ex_all_uppercased`: are all characters of the token upper-case?
* `ex_mixed_case`: among the non-initial characters, are there both upper case and lower case characters?
* `ex_internal_period`: does the token include a period (full stop) that is non-initial and non-final?
* `ex_non_letter`: does the token include a character that is not a letter and not a mark (according to Unicode definitions)?
* `ex_digits`: does the character contain digits?
* `ex_long_token`: is the token longer than a threshold (default=8 characters)?
* `ex_contains_latin`: does the token include Latin characters?
* `ex_contains_ethiopic`: does the token include Ethiopic characters?

### Token-Level Features

* `ex_title`: is the preceding token a title? Note that in Somali, titles are not used before personal names.
* `ex_head_org`: is the token a head word for an organization?
* `ex_head_loc`: is the token a head word for a location or does it include such a word?
* `ex_head_gpe`: is the token a head word for a geopolitical entity or does it include such a word?
* `ex_prep_from`: is the token, or does the token include, a preposition meaning 'from'
* `ex_prep_in`: is the token, or does the token include, a preposition meaning 'in'

### Gazetteer Features

* `ex_b_gaz, LOC`: token is first token of LOC in gazetteer
* `ex_b_gaz, GPE`: token is first token of GPE in gazetteer
* `ex_b_gaz, ORG`: token is first token of ORG in gazetteer
* `ex_b_gaz, PER`: token is first token of PER in gazetteer
* `ex_i_gaz, LOC`: token is non-initial token of LOC in gazetteer
* `ex_i_gaz, GPE`: token is non-initial token of GPE in gazetteer
* `ex_i_gaz, ORG`: token is non-initial token of ORG in gazetteer
* `ex_i_gaz, PER`: token is non-initial token of PER in gazetteer
* `ex_o_gaz`: token is not in a gazetteer entry
