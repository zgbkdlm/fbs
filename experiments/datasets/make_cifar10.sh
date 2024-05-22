wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
python make_cifar10.py
rm -rf ./cifar-10-batches-py
rm cifar-10-python.tar.gz
