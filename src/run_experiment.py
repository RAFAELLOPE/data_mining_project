import os
import sys
sys.path.append(os.path.abspath('.\\'))

from src.trainer.trainer import Experiment
from src.config.config import Config


if __name__ == "__main__":
    config = Config()
    experiment = Experiment(config)
    print('Running Experiment ...')
    print('\n')
    experiment.run()
    print('\n')
    print('Experiment Finished!!')
