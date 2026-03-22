#!/usr/bin/env python3
"""Quick demo / smoke test for metapad_preprocess.py"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from metapad_preprocess import build_pipeline, process_text

nlp = build_pipeline("en_core_web_sm")

sample_texts = [
    "Barack Obama was born in Honolulu, Hawaii on August 4, 1961. "
    "He served as the 44th President of the United States from 2009 to 2017.",

    "Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California. "
    "The company released the iPhone in January 2007, which revolutionized the smartphone industry.",

    "The European Central Bank announced a new monetary policy on March 15, 2023. "
    "Christine Lagarde stated that interest rates would remain at 3.5 percent.",
]

for i, text in enumerate(sample_texts):
    print(f"\n{'='*72}")
    print(f"[Raw Text {i+1}]")
    print(text)
    result = process_text(nlp, text, fmt="both")
    print(f"\n[XML Format]")
    for line in result["xml"]:
        print(f"  {line}")
    print(f"\n[Dollar Format (lowercased, $TYPE tags preserved)]")
    for line in result["dollar"]:
        print(f"  {line}")
