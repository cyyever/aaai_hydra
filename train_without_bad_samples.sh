python3 train_without_bad_samples.py --dataset_name MNIST --model_name LeNet5 --epochs 20 --contribution_dir final_models/MNIST/LeNet5/92480fa8-ee34-415e-ab82-be227f9110b7/randomized_model
# python3 random_train_with_hypergradient.py --dataset_name FashionMNIST --model_name LeNet5 --cache_size 4096 --epochs 20 --random_percentage 0.8
# python3 random_train_with_hypergradient.py --dataset_name CIFAR10 --model_name DenseNet40 --cache_size 500 --epochs 20 --random_percentage 0.8
