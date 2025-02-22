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

**_num_epochs_** parameter means count of train epochs.

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

**mbatch_size** parameter means minibatch size.

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

**type** parameter means type of optimizer. The two optimizers are available now: SGD and Adam.
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