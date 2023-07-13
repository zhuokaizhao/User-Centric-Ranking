# User-Centric Ranking (UCR)

This is the official implementation of our paper [Breaking the Curse of Quality Saturation with User-Centric Ranking](https://arxiv.org/abs/2305.15333). The code within this repository relies heavily on [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch). We thank the authors for their contributions to the open-sourced community.

## Data

MovieLens data used in this experiment as well as pre-processed user-centric features can be found [here](https://drive.google.com/drive/folders/1INVyJTy1pWZuQHR6UeH9BkRrfIc7BSim?usp=sharing).

## Set up environment

Use the included `setup_env.sh` to set up your environment by running

```sh
    sh setup_env.sh
```

## Create User-Centric Features

As mentioned before, pre-processed user-centric features can be downloaded from [here](https://drive.google.com/drive/folders/1INVyJTy1pWZuQHR6UeH9BkRrfIc7BSim?usp=sharing). But to generate the features, you can run `process_data.py` with the following options.

`--data_type`: Choose between '1M', '10M', or '20M' which each corresponds to one variant of the MovieLens datasets. '25M' for the MovieLens25M is currently not supported unfortunately.

`--input_dir`: Directory of the root that contains the data, for example, it could be `./data/ml-1m`.

`--output_dir`: Directory that will be used to save the generated features.

`--num_process`: For multi-process purposes.

`-v`, `--verbose`: Verbosity.

An example command for generating user-centric features can be

```shell
python process_features.py --data_type 10M --input_dir ./data/ml-1m/ --output_dir ./data/ --num_process 10 -v
```

## Run Experiment

To run the experiment, simply run the main file `main.py` with the following options

`--mode`: Choose either 'train' or 'test'.

`--model_name`: Name of the model, choose between 'DIN', 'DIEN', or 'DIFM'.

`--model_type`: Type of pooling operation, choose between 'sum' and 'attention'. Check Table 3 in our paper for more details.

`--data_type`: Same as before, choose between '1M', '10M' or '20M'.

`--feature_type`: Choose between 'IC' (item-centric) and 'UC' (user-centric).

`--train_dir`: Directory of the training features (if in 'train' mode). Running `process_features.py` or downloading pre-process features should have this ready. An example could be './data/splitted_features_10m/train'.

`--val_dir`: Directory of the validation features (if in 'train' mode). An example could be './data/splitted_features_10m/test'.

`--test_dir`: Directory of the test features (if in 'test' mode). An example could be './data/splitted_features_10m/test'.

`--output_model_dir`: Directory for trained model checkpoints.

`--input_model_path`: Path of the existing model, required if continuing training or doing testing.

`--output_hist_dir`: Directory of the training history.

`--num_epoch`: Number of training epochs.

`--batch_size`: Batch size.

`--save_freq`: Number of epochs when a model checkpoint is saved.

`-v`, `--verbose`: Verbosity.

An example of training command might look like:

```sh
python main.py --mode train --model_name DIN --model_type sum --data_type 10M --feature_type UC --train_dir ./data/splitted_features_10M/train --val_dir ./data/splitted_features_10M/test --output_model_dir ./models --output_hist_dir ./history --num_epoch 100 --batch_size 64 --save_freq 10 -v
```

An example of test command might look like:

```sh
python main.py --mode test --model_name DIN --model_type sum --data_type 10M --feature_type UC --test_dir ./data/splitted_features_10M/test --input_model_path ./models/trained_model.pt --batch_size 128 -v
```

## Reference

If this code helps your research, please kindly cite our paper. Thank you!

```latex
@article{zhao2023breaking,
  title={Breaking the Curse of Quality Saturation with User-Centric Ranking},
  author={Zhao, Zhuokai and Yang, Yang and Wang, Wenyu and Liu, Chihuang and Shi, Yu and Hu, Wenjie and Zhang, Haotian and Yang, Shuang},
  journal={arXiv preprint arXiv:2305.15333},
  year={2023}
}
```
