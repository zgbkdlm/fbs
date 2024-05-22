import numpy as np
import jax
from PIL import Image

images128 = np.zeros((30000, 128, 128, 3), dtype='float32')

for i in range(30000):
	filename = f'./data128x128/{i + 1:05}.jpg'
	img = np.asarray(Image.open(filename))
	images128[i] = img / 255.
images128[images128 > 1.] = 1.

images64 = jax.image.resize(images128, (30000, 64, 64, 3), method='linear')


np.save('celeba_hq128', images128)
np.save('celeba_hq64', images64)

