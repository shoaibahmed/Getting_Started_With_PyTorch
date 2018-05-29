# PyTorch Deformable FCN

Implementation of Deformable FCN with PyTorch.

```
python train.py -t --rootDir ./data -o ./output -e 100 -d 100
```

Where -t is the flag for training, --rootDir specifies the path to the dataset, -o specifies the output directory, -e specifies the number of epochs for training and -d is the number of steps between consecutive print statements of the loss.
There are many other options as well, which can be checked out using the --help flag.

```
python train.py --help
```
