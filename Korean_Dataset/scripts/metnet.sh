python train.py --model="metnet" --dataset_dir="$HOME/AMAL/project/KoMet-Benchmark-Dataset/data/nims" --device=0 --seed=2 --input_data="gdaps_kim" \
                --num_epochs=50 --normalization \
                --rain_thresholds 0.1 10.0 \
                --interpolate_aws \
                --intermediate_test \
                --custom_name="metnet_test_3"