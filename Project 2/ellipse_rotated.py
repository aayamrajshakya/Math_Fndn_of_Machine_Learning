#https://matplotlib.org/2.2.2/gallery/shapes_and_collections/ellipse_rotated.html
#---------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

delta = 45.0  # degrees
angles = np.arange(0, 180, delta)
print('Angles =',angles)

ells = [Ellipse((1, 1), 4, 2, a) for a in angles]

p = plt.subplot(111, aspect='equal')

for e in ells:
    e.set_clip_box(p.bbox)
    e.set_alpha(0.2)
    p.add_artist(e)

plt.xlim(-1, 3)
plt.ylim(-1, 3)

plt.show()
