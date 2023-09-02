import pickle
import time
from hyperopt import fmin, tpe, hp, STATUS_OK

def objective(in_dict):
    return {'loss': in_dict['x'] ** 2 - in_dict['y'] ** 2, 'status': STATUS_OK }

space = {
        'x': hp.uniform('x', -10, 10),
        'y': hp.uniform('y', -10, 10),
        }

best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=1500)

print(best)