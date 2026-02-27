import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOSS_PATTERN = re.compile(r"losses_distributed\s+\[.*\((\d+),\s*([\d.]+)\)\]")