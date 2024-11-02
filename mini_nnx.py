import jax, jax.numpy as jnp
from flax import nnx
import optax, cv2, wandb, numpy as np
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from tqdm.auto import tqdm


# confirm devices
print(f'JAX devices: {jax.local_devices()}')

split: int = 10000
hfdata = load_dataset('uoft-cs/cifar10', split='train', streaming=True).take(split)


def read_image(img, img_size: int = 32):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


class ImageData(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return split

    def __iter__(self):
        for sample in self.dataset:
            image = sample["img"]
            image = read_image(image)

            image = jnp.array(image)
            label = jnp.array(sample["label"])

            yield image, label  # sample["label"]#label


def jax_collate(batch):
    images, labels = zip(*batch)
    batch = (jnp.array(images), jnp.array(labels))
    batch = jax.tree_util.tree_map(jnp.array, batch)
    return batch


traindata = ImageData()
train_loader = DataLoader(traindata, batch_size=32, collate_fn=jax_collate)

xc = next(iter(train_loader))

# xc[0].shape, xc[1].shape


class MobileBlock(nnx.Module):
    def __init__(self, inchan, outchan, stride=1):
        self.depthwise_conv = nnx.Sequential(
            nnx.Conv(
                inchan, inchan, kernel_size=3, 
                strides=stride, padding=1, feature_group_count=inchan
            ),
            nnx.BatchNorm(inchan)
        )
        self.pointwise = nnx.Sequential(
            nnx.Conv(inchan, outchan, kernel_size=1, strides=1, padding=0),
            nnx.BatchNorm(outchan)
        )

    def __call__(self, x_img):
        x = nnx.relu(self.depthwise_conv(x_img))
        x = nnx.relu(self.pointwise(x))
        
        return x


class JaxMobilenet(nnx.Module):
    def __init__(self, num_classes=10):
        self.inputconv = nnx.Conv(3, 32, kernel_size=3, strides=2, padding=1)
        self.batchnorm = nnx.BatchNorm(32)
        
        self.convlayer = nnx.Sequential(
            MobileBlock(32, 64),
            MobileBlock(64, 128, stride=2),
            MobileBlock(128, 256, stride=2),    
        )
        self.linear_fc = nnx.Linear(256, num_classes)
    
    def __call__(self, x_img: jax.Array):
        x = self.batchnorm(self.inputconv(x_img))
        x = nnx.relu(x)
        x = self.convlayer(x)
        x = x.view(-1, 256)
        
        return x #nnx.softmax(x, axis=1)

cnn_model = JaxMobilenet()
nnx.display(cnn_model)

learn_rate = 1e-4
optimizer = nnx.Optimizer(cnn_model, optax.adamw(learning_rate=learn_rate))
metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss')
)

def loss_func(batch, model=cnn_model):
    image, label = batch
    logits = model(image)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()

    return loss, logits


def wandb_logger(key: str, model, project_name, run_name):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(project=project_name, name=run_name)
    wandb.watch(model)


@jax.pmap(axis_name="batch")
@jax.pmap(axis_name="model")
@nnx.jit
def train_step(model, optimizer, metrics: nnx.MultiMetric, batch):
    gradfn = nnx.value_and_grad(loss_func, has_aux=True)
    (loss, logits), grads = gradfn(batch, model)
    metrics.update(loss=loss, logits=logits, labels=batch[1])
    optimizer.update(grads)
    
    acc = metrics.compute()['accuracy']
    return loss, acc

def trainer(model=cnn_model, optimizer=optimizer, train_loader=train_loader):
    epochs = 10
    wandb_logger(key=None, model=model, project_name='play_jax', run_name='mobilecifar-256c')
    
    for epoch in tqdm.range(epochs):
        for step, batch in tqdm(enumerate(train_loader)):
            
            train_loss, accuracy = train_step(model, optimizer, metrics, batch)
            
            print(f'step {step}, loss-> {train_loss.item():.4f}, acc {accuracy.item():.4f}')
            
            wandb.log({'loss': train_loss, 'accuracy':accuracy})
            
        print(f'epoch {epoch}, train loss mode')
