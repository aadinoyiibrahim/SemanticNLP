"""
Install packages for the project
"""

import os

packages = ["matplotlib", "pandas", "numpy", "nltk", "torch", "transformers"]

for st_ in packages:
    os.system("pip install " + st_)
