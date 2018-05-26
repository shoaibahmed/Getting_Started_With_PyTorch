# PyTorch LSTM Example

A sample LSTM network for classification with PyTorch.

```
python train.py -t --rootDir ./data -o ./output -b 50 -e 100 -d 100
```

Where -t is the flag for training, --rootDir specifies the path to the dataset, -o specifies the output directory, -b specifies the batch size, -e specifies the number of epochs for training and -d is the number of steps between consecutive print statements of the loss.
There are many other options as well, which can be checked out using the --help flag.

```
python train.py --help
```
