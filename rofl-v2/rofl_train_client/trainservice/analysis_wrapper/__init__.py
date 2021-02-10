

import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
analysis_path = os.path.join(dir_path, '../../../../adversarial-framework')
sys.path.insert(0, analysis_path)

from .analysis_client import AnalysisClientWrapper