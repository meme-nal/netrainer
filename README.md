# netrainer

It's a utility that provides a convenient way to train neural networks.

## Installation

### Requirements

This program is based on two C++ libraries: __libtorch__ - the C++ API for Pytorch and __nlohmann/json__. Make sure that you already have these libraries installed.

You can install these libraries from here:
- [libtorch](https://github.com/pytorch/pytorch)
- [nlohmann/json](https://github.com/nlohmann/json)

Clone the repo

```shell
git clone https://github.com/meme-nal/netrainer.git
cd netrainer
```

Build

```shell
mkdir build
cd build
cmake -S ../src -B .
make
```

## Usage

Example of usage

```shell
./netrainer -c <path_to_net_config> -b <path_to_batches>
```

You can find the rules on how to write configs for the program [here](./docs/doc.md). Note that batches must have a strictly defined configuration. You can find script for creating batches [here]().
