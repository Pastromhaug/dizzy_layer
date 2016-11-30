import numpy as np
from gen_data import gen_data, gen_epochs

epochs = gen_epochs(3,48,4,3)
for i, epoch in enumerate(epochs):
    print(epoch)
