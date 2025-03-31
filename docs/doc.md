# Usage documenation

Program takes .json configuration file as an argument. Generally file consists of 4 main parts: **_common options_**, **_optimizer options_**, **_label generator_** and **_nn architecture_**.

end-to-end example of nn configuration file:

```json
{
  "num_epochs": 500,
  "to_save_path": "/home/me_me/models/simple_model",
  "mbatch_size": 512,
  "device": "cpu",

  "optimizer": {
    "type": "Adam",
    "lr": 0.001
  },

  "label_generator": {
    "type": "default",
    "rules": "path_to_rules.json",
    "label_smoothing": {
      "to_use": false,
      "eps": 0.001
    }
  },

  "arch": {
    "Flatten": {
      "type": "flatten",
      "input": "main_data",
      "output": "Flatten_out"
    },

    "fc1": {
      "type": "dense",
      "input": "Flatten_out",
      "output": "fc1_out",
      "shape": [1200, 6000],
      "nonlinearity": "ReLU",
      "bias": true
    },

    "fc2": {
      "type": "dense",
      "input": "fc1_out",
      "output": "fc2_out",
      "shape": [6000, 2400],
      "nonlinearity": "ReLU",
      "bias": true
    },

    "fc3": {
      "type": "dense",
      "input": "fc2_out",
      "output": "fc3_out",
      "shape": [2400, 1200],
      "nonlinearity": "Linear",
      "bias": true
    },

    "Reshape": {
      "type": "reshape",
      "input": "fc3_out",
      "output": "prediction",
      "shape": [1, 200, 6]
    },

    "final": {
      "type": "loss",
      "cost": "MSE"
    }
  }
}
```

## Common options

**_num_epochs_** - number of training epochs.

```json
{
  "num_epochs": 500
}
```

---

**_to_save_path_** parameter means in what directory to save trained model. Program saves model every epoch.

```json
{
  "to_save_path": 500
}
```

---

**mbatch_size** - minibatch size.

```json
{
  "mbatch_size": 512
}
```

---

**device** parameter means on what device to train model. There are two possible values: cpu and gpu. 

```json
{
  "device": "cpu"
}
```


## Optimizer options

**type** - type of optimizer. \
\
Available optimizer types:
- SGD
- Adam

**lr** parameter means value of learning rate.

```json
{
  "optimizer": {
    "type": "Adam",
    "lr": 0.001
  }
}
```

## Label generator

**type** - type of label generator. \
Available label generator types:
- default - does not change the label in batch. Commonly used in regression tasks. Label smoothing is not used.
- ClassLabelGenerator - performs one hot encoding over label in batch. If there are K classes, then only the class number _k_ is added to the label in batch. Commonly used in classification tasks. Label Smoothing is used. 

**rules** - path to json with additional information about custom label creation. Used in ClassLabelGenerator.

rules.json for simple classification tasks:
```json
{
  "class_0": 0,
  "class_1": 1,
  "class_2": 2
}
```

**label_smoothing** - trick that smooths one hot label.

## Architecture

Conventionally there are several types of layers. Common layers, auxiliary layers and loss layers.

### Common layers

#### Dense layer

Or fully connected layer. It performs a linear transformation over input tensor. \
\
**type** - type of layer. \
**input** - input tensor. \
**output** - output of layer. \
**shape** - shape of weights matrix. \
**nonlinearity** parameter means what type of activation function will be applied to layer output. \
\
Available nonlinearities:
- Linear
- Sigmoid
- Swish
- Tanh
- ReLU
- ELU
- LeakyReLU
- SELU
- CELU
- GELU
- Softmax

**bias** - use bias or not. 

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "dense",
      "input": "inputTensorName",
      "output": "outputTensorName",
      "shape": [100, 10],
      "nonlinearity": "Linear",
      "bias": true
    }
  }
}
```

---

#### Conv layer

**type** - type of layer. \
**input** - input tensor. \
**output** - output of layer. \
**in_channels** - number of input channels. \
**out_channels** - number of output channels. \
**kernel** - shape of conv filter. \
**nonlinearity** - parameter means what type of activation function will be applied to layer output. \
**bias** - use bias or not. \
**stride** - shape of stride. First number - step along X axis, Second number - step along Y axis.\
**padding** - shape of padding. First number - number of values to be added along the X axis, Second number - number of values to be added along the Y axis. \
**dilation** - shape of dilation. Default is [1, 1] \
**groups** - number of groups Default is 1. 

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "conv",
      "input": "inputTensorName",
      "output": "outputTensorName",
      "in_channels": 1,
      "out_channels": 8,
      "kernel": [20, 6],
      "nonlinearity": "ReLU",
      "bias": true,
      "stride": [1, 1],
      "padding": [0, 0],
      "dilation": [1, 1],
      "groups": 1
    }
  }
}
```

---

#### Pooling layers

**type** - type of layer \
**subtype** - max pooling or avg pooling \
**input** - input tensor \
**output** - output tensor \
**kernel** - shape of pooling filter \
**nonlinearity** - parameter means what type of activation function will be applied to layer output. \
**stride** - shape of stride. First number - step along X axis, Second number - step along Y axis.

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "pooling",
      "subtype": "max",
      "input": "inputTensorName",
      "output": "outputTensorName",
      "kernel": [3, 3],
      "nonlinearity": "ReLU",
      "stride": [2, 2]
    }
  }
}
```

### Auxiliary layers

#### Flatten layer

**type** - type of layer. \
**input** - input tensor. \
**output** - output of layer.

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "flatten",
      "input": "inputTensorName",
      "output": "outputTensorName"
    }
  }
}
```

---

#### Reshape layer

**type** - type of layer. \
**input** - input tensor. \
**output** - output of layer. \
**shape** - shape of output tensor.

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "reshape",
      "input": "inputTensorName",
      "output": "outputTensorName",
      "shape": [1, 50, 50]
    }
  }
}
```

---


### Special layers

Don't forget about inference when using special layers!

#### Batch Normalization layer

**type** - type of layer. \
**input** - input tensor. \
**output** - output tensor. \
**channels** - count of input channels or input features.\
**dims** - 1d or 2d batch normalization. 1d used after dense layer. 2d can be used before and after conv layer.

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "bn",
      "input": "inputTensorName",
      "output": "outputTensorName",
      "channels": 32,
      "dims": 2
    }
  }
}
```

---

#### Dropout layer

**type** - type of layer. \
**input** - input tensor. \
**output** - output tensor. \
**prob** - probability of disabling specific neuron.\
**dims** - 1d or 2d dropout. 1d used after dense layer. 2d can be used before and after conv layer.

```json
{
  "arch": {
    "CustomLayerName": {
      "type": "dropout",
      "input": "inputTensorName",
      "output": "outputTensorName",
      "prob": 0.5,
      "dims": 2
    }
  }
}
```

---

### Loss layers

The last layer is used to calculate loss while training. Note that this layer has specific **_final_** name. \
\
**type** - type of layer. \
**cost** - type of cost function. \
\
Available cost functions:
- MAE
- MSE
- CrossEntropy

```json
{
  "arch": {
    "final": {
      "type": "loss",
      "cost": "MSE"
    }
  }
}
```

