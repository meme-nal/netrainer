# Usage documenation

Program takes .json configuration file as an argument. Generally file consists of 3 main parts: **_common options_**, **_optimizer options_** and **_nn architecture_**.

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
      "nonlinearity": "Linearity",
      "bias": true
    }
  }
}
```

---

#### Conv layer

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

