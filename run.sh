#!/bin/sh

echo "Generating output for boolean retrieval ..."
python3 run_model1.py "$1"
echo "Generating output for tf-idf ..."
python3 run_model2.py "$1"
echo "Generating output for BM25 ..."
python3 run_model3.py "$1"
