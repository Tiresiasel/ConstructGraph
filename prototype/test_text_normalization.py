#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small test to verify the current normalization keeps the sample text intact.
Run: python scripts/test_text_normalization.py
"""
import re

def normalize_math_inline(text: str) -> str:
    if text is None:
        return ''
    s = str(text)
    # Step 4: Fix specific LaTeX-like patterns FIRST
    s = re.sub(r'/SL([a-zA-Z]+)', r'\\\1', s)
    s = re.sub(r'/lparenori', '(', s)
    s = re.sub(r'/rparenori', ')', s)
    s = re.sub(r'/commaori', ',', s)
    s = re.sub(r'/lbracketori', '[', s)
    s = re.sub(r'/rbracketori', ']', s)
    # Step 5: Fix common abbreviations ONLY
    s = re.sub(r'\bi\.\s*e\.', 'i.e.', s, flags=re.IGNORECASE)
    s = re.sub(r'\be\.\s*g\.', 'e.g.', s, flags=re.IGNORECASE)
    s = re.sub(r'\betc\.', 'etc.', s, flags=re.IGNORECASE)
    # Step 6: ONLY camelCase split + whitespace collapse (do not change punctuation spacing)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s)
    # Step 7: Ensure e.g., i.e., with comma variants
    s = re.sub(r'e\.\s*g\.\s*,', 'e.g.,', s)
    s = re.sub(r'i\.\s*e\.\s*,', 'i.e.,', s)
    return s

def check_sample(sample: str, expected: str) -> None:
    out = normalize_math_inline(sample)
    print('INPUT  :', sample)
    print('OUTPUT :', out)
    print('EXPECT :', expected)
    assert out == expected, 'Normalization changed the text unexpectedly.'

if __name__ == '__main__':
    # Sample per your screenshot intent (without parentheses)
    sample1 = 'e.g., better professionals, brand equity that affects competitive outcomes.'
    expected1 = 'e.g., better professionals, brand equity that affects competitive outcomes.'
    check_sample(sample1, expected1)

    # With parentheses (as appears in DB definitions)
    sample2 = '(e.g., better professionals, brand equity) that affects competitive outcomes.'
    expected2 = '(e.g., better professionals, brand equity) that affects competitive outcomes.'
    check_sample(sample2, expected2)

    # Ensure hyphenated words are preserved
    sample3 = 'Firm-specific capability and e.g., clear examples.'
    expected3 = 'Firm-specific capability and e.g., clear examples.'
    check_sample(sample3, expected3)

    print('\nAll tests passed.')
