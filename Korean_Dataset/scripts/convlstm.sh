python train.py --model="convlstm" --dataset_dir="$HOME/AMAL/project/KoMet-Benchmark-Dataset/data/nims" --device=0 --seed=2 --input_data="gdaps_kim" \
                --num_epochs=50 \
                --rain_thresholds 0.1 10.0 \
                --interpolate_aws \
                --intermediate_test \
                --custom_name="convlstm_test_test"