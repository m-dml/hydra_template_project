# Hydra Lightning Template for Structured Configs
Template for creating projects with pytorch-lightning and hydra.
What is different to https://github.com/ashleve/lightning-hydra-template ? Some ideas are taken from there, but the big difference is that my template uses structured configs.

# How to use this template?
Create your own project on GitHub with this template by clicking the
[Use this template](https://github.com/m-dml/hydra_template_project/generate) button.

You now have to only add your own dataloader, dataset, model, optimizer and loss and you should be ready to go.
To see if you have all modules installed and everything works fine, you should run the unit tests!

# How to add my own module?
For this tutorial it is expected that you already know pytorch (and best also some pytorch-lightning). If you don't
know hydra that should be fine, but definitely check out [their docs](https://hydra.cc/).

If you encounter any problems have a look at the `my_simple_model` branch of this repo, where I played through
this complete tutorial. So you can find all files there.

Lets explore how to use hydra and this template by showcasing how one would add a simple own CNN to this repo.
For the tests I used MNIST as dataset so we will just continue using that. But if you know how to write a
pytorch-lightning Dataloader and a torch Dataset it should be just as easy to replace them after this small tutorial.

To add our own model we have to do the following steps:
1. in the folder `src/models` we create a new file containing our torch model (a torch.nn.Module).
2. Add the model in the hydra config library by adding it to the `src/lib/model.py` file.
3. Register the model in the hydra global-config-register by following the pattern in `src/lib/config.py` and creating
a new entry there.
4. (Optional) Create a `yaml` file for the model. This makes sense if the model is used with a lot of different
settings. So we can give those settings individual names, which makes them easier to call.
5. Add an experiment using that model

### 1. Creating the simplest model:
Create the file `src/models/my_simple_model.py` with the following content:

```python
import torch.nn as nn
import torch.nn.functional as F


class MySimpleModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(MySimpleModel, self).__init__()

        # When the image enters the net at conv1 it has a size of 28x28x1, because there is a single color channel
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # Since we are using padding the size of the image does not change after the conv layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # due to the maxpooling shape and stride our image is now 14x14
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)
        # still 14x14
        # We will again use maxpool so now it is 7x7
        self.fully_connected = nn.Linear(16 * 7 * 7, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = x.flatten(start_dim=1)  # To use a fully connected layer in the end we need to have a 1D array
        x = self.fully_connected(x)
        return F.softmax(x)  #  we apply a softmax here to return probabilities between 0 and 1

```

### 2. Add the model to the lib:
Change the file `src/lib/model.py` to add our model there. Just add the following lines:

```python
@dataclass
class MySimpleModelLib:
    _target_: str = "src.models.my_simple_model.MySimpleModel"
    input_channels: int = 1
    num_classes: int = 10
```

A few pittfalls to avoid are:
* Do not forget to decorate your class with @dataclass !
* do not forget to specify the type !
* Have a look at other lib files to see how to implement `None` as default and use the `Any` type.
* do not forget any inputs to the actual model (like forget the parameter `input_channels`) because you will never be
able to override the input channels from outside the source code.

### 3. Register the model in hydra:
For hydra to know about your model, you have to register it. We do this in the file `src/lib/config.py`. All we have to
do here is adding 2 lines.

1. We have to import the library model. So at the imports we add:
```python
from src.lib.model import MySimpleModelLib
````

2. Register the model by using the hydra ConfigStore. Best keep the code clean, so find the section where the models are
defined and add:
```python
cs.store(name="my_simple_model_base", node=MySimpleModelLib, group=model_group)
````
I like to append the `_base` her to later distinguish between the yaml-config and the structured-config. If you want
to know more about this you will probably have to read the hydra documentation.

### 4. Add a yaml config file:
This step is not necessary. We could already use our model in hydra now, which would at this point go under the name
`my_simple_model_base`. But for the sake of completion lets create a yaml config as well.

For this we will have to create this file: `conf/model/my_simple_model.yaml`

The content of this file should be

```yaml
defaults:
  - my_simple_model_base  # this is the name of the registered model that we would like to extend
  - _self_  # adding this BELOW!! the registered name means, that everything in this yaml file will override the defaults

# you can only specify values here that are also in the registered model (src/lib/model/MySimpleModelLib)
num_classes: 10
input_channels: 1
````

If you want, you can of course drop the comments.

Why did we create this config file? Lets say you would like to also have t he same model, but with 3 input channels when
you do predictions on colored images. All you would have to do is either change the value `input_channels: 3` of the
file `conf/model/my_simple_model.yaml`. But if you want to give it a distiguishable name (which makes sense for more
complex usecases) then you can just create another file `conf/model/my_simple_model_rgb.yaml` for example, which has
the content

```yaml
defaults:
  - my_simple_model_base
  - _self_

num_classes: 10
input_channels: 3  # <- this is the only thing that changed
````

Now you could from a command line very easily switch between the 2 configs without remembering any specific numbers.

### 5. Add an experiment using that model:
There are 2 ways to use your model now in a training run.

1. From the command line:
All you have to do is keep everything with the defaults and just exchanging the model from the command line using hydras
command line interface:
```bash
python main.py model=my_simple_model
```
or
```bash
python main.py model=my_simple_model_rgb
```
or if you did not create the yaml-file:
```bash
python main.py model=my_simple_model_base
```
From the command line we could also specify different inputs to our model:
```bash
python main.py model=my_simple_model_base model.input_channels=3
```

2. We can create an experiment using this model. This definitely is preferable when the setups get more complex.
For this, we have to create a new yaml file in the experiment folder. So lets create the file
`conf/experiment/my_simple_model_experiment.yaml` with the following content:


```yaml
# @package _global_

defaults:
  - override /lightning_module: default
  - override /datamodule: mnist
  - override /datamodule/dataset: mnist
  - override /loss: nll_loss
  - override /datamodule/train_transforms: no_transforms
  - override /datamodule/valid_transforms: no_transforms
  - override /model: my_simple_model  # <- this is the line where we add our own model to the experiment
  - override /optimizer: sgd
  - override /loss: nll_loss
  - override /strategy: null
  - override /logger/tensorboard: tensorboard
  - override /callbacks/checkpoint: model_checkpoint
  - override /callbacks/early_stopping: early_stopping
  - override /callbacks/lr_monitor: lr_monitor

  - override /hydra/launcher: local
  - _self_

output_dir_base_path: ./outputs
random_seed: 7
print_config: true
log_level: "info"

trainer:
  fast_dev_run: false
  num_sanity_val_steps: 3
  max_epochs: 3
  gpus: 0
  limit_train_batches: 3
  limit_val_batches: 3

datamodule:
  num_workers: 0
  batch_size: 4

```
Most settings here are the same as in the defaults, which are specified in `conf/config.yaml` but for this tutorial I
think explicit is easier to understand the implicit.

To use the experiment we run our model with
```bash
python main.py +experiment=my_simple_model_experiment
```

Again we can also change all set values from the command line
```bash
python main.py +experiment=my_simple_model_experiment datamodule.num_workers=20
```

It should be easy now to follow the same steps to include your own datamodule, dataset, transforms, optimizers or
whatever else you might need.
