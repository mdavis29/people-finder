import os
from tqdm import tqdm
n_batches = 50
for i in tqdm(range(n_batches)):
    os.system('python3 model.py')

print('build_model.py completed')
