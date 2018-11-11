The repository is consited of four parts

1. utils/...   			This folder save helper scripts for data processing.
	-> data_generater.py   	Data generater for Keras model input
	-> load_data.py		Helper function to load CIFAR10
	-> random_eraser.py	The script for Random Erasing augmentation

2. traditional_methods/... 	This folder save the scripts for BOW and Fisher Vector
	-> bag_of_word.py	Run this scripts in the ROOT folder of this project to use traditional model
	-> fisher_vector.py	Helper function to get fisher vectors from features

3. deep_methods/...		This folder save the defination of deep model architectures
	-> models.py		Contains all deep model architectures I used

4. cifar10.py			Run this script to use deep learning models