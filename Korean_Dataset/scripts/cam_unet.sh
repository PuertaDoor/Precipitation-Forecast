python train.py --model="cam_unet"  --dataset_dir="$HOME/AMAL/project/KoMet-Benchmark-Dataset/data/nims" --device=0 --seed=0 --input_data="gdaps_kim" \
                --num_epochs=20 --normalization \
                --rain_thresholds 0.1 10.0 \
                --interpolate_aws \
                --intermediate_test \
                --custom_name="cam_unet_test_run"