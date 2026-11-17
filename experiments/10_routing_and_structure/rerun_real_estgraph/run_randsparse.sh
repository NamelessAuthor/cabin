#!/bin/bash
cd "$(dirname "$0")"
python run_real_estgraph.py --workers 5 --configs mlp_randsparse --tasks MiceProtein analcatdata_authorship mfeat-factors mfeat-fourier mfeat-karhunen mfeat-zernike ozone-level-8hr semeion spambase splice
python run_real_estgraph.py --workers 5 --configs mlp_randsparse_ens5 --tasks MiceProtein analcatdata_authorship mfeat-factors mfeat-fourier mfeat-karhunen mfeat-zernike ozone-level-8hr semeion spambase splice
python run_real_estgraph.py --workers 5 --configs mlp_randsparse
python run_real_estgraph.py --workers 5 --configs mlp_randsparse_ens5
