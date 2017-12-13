#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

PATTERNS = [
    (r'([aeiou])(\1)', r'\1'),
    (r'(b|c|ch|d|dh|f|g|h|j|k|l|m|n|ny|p|ph|q|r|s|sh|t|v|w|x|y|z)\1', r'\1'),
    (r'ph', r'p'),
    (r'q', r'k'),
    (r'x', r't'),
    (r'c([^h]|\b)', r'ch\1'),
    (r'ai', r'ayi'),
    (r's(b|c|ch|d|dh|f|g|h|j|k|l|m|n|ny|p|ph|q|r|s|sh|t|v|w|x|y|z)', r'f\1'),
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
