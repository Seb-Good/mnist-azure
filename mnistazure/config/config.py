"""
config.py
By: Sebastian D. Goodfellow, Ph.D., 2019
"""

# 3rd party imports
import os

# Root working directory
WORKING_DIR = (
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__)
            )
        )
    )
)

# Projects path
DATA_PATH = os.path.join(WORKING_DIR, 'data')
