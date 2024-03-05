#!/bin/bash
for i in $(seq 1 20); do
  clean_file_name="clean_resnetmod_tinyimagenet_0"$(echo $RANDOM | cut -b2)""$(echo $RANDOM | cut -b2)""$(echo $RANDOM | cut -b2)".pth";
  ln -s "/home/hegedusi/backdoor_models/ULP/tiny-imagenet/train_clean/"$clean_file_name $clean_file_name;
  poisoned_file_name="poisoned_resnetmod_tiny-imagenet_0"$(echo $RANDOM | cut -b2)""$(echo $RANDOM | cut -b2)""$(echo $RANDOM | cut -b2)".pt";
  ln -s "/home/hegedusi/backdoor_models/ULP/tiny-imagenet/train_poisoned/"$poisoned_file_name $poisoned_file_name
done