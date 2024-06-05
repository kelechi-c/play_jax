import flax
from flax import nnx, linen as lnn

class Convnet(lnn.module):
    
    @lnn.compact
    def __call__(self, img):
        x = lnn.Conv(features=32, kernel_size=(3, 3))(img)
        x = lnn.relu(x)
        x = lnn.avg_pool(x, window_shape=())