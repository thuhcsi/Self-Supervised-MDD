#!/bin/bash
# rsync -a ./Self-Supervised-MDD/ ./fairseq/fairseq/
. ./path.sh

config_name=mean_teacher_ft_sample_focal_ctc_gamma4
checkpoint_path=/mnt/d/projects/Self-Supervised-MDD/results/finetune_base_t/checkpoint_best.pt


stage=2
stop_stage=3

timit_dir=/mnt/d/data/TIMIT
raw_l2_arctic_dir=/mnt/d/data/L2-Arctic
l2_arctic_dir=/mnt/d/data/L2-Arctic_ds
all_data_dir=/mnt/d/data/

g2p_dir=./g2p

data_dir=/mnt/d/projects/Self-Supervised-MDD/data
trans_path=/mnt/d/projects/Self-Supervised-MDD/data/train_u/trans_g2p
phoneme_map='60-40'

output_dir=/mnt/d/projects/Self-Supervised-MDD/results


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Preprocess data..."
    python3 ./Self-Supervised-MDD/scripts/timit_downsampling.py --raw_l2_arctic_dir $raw_l2_arctic_dir --output_dir $l2_arctic_dir

    Self-Supervised-MDD/scripts/timit_data_prep.sh $timit_dir $phoneme_map || exit 1;
    echo "timit prep done"

    python Self-Supervised-MDD/scripts/l2arctic_prep.py --l2_path=$l2_arctic_dir --save_path=${data_dir}/l2_train  
    python Self-Supervised-MDD/scripts/l2arctic_prep.py --l2_path=$l2_arctic_dir --save_path=${data_dir}/l2_dev  
    python Self-Supervised-MDD/scripts/l2arctic_prep.py --l2_path=$l2_arctic_dir --save_path=${data_dir}/l2_test
    mv ${data_dir}/l2_dev ${data_dir}/dev  
    mv ${data_dir}/l2_test ${data_dir}/test
    Self-Supervised-MDD/scripts/timit_l2_merge.sh ${data_dir}/train_timit ${data_dir}/l2_train ${data_dir}/train
    python Self-Supervised-MDD/scripts/trans_prep_g2p.py --l2arctic_dir=$l2_arctic_dir\
     --timit_dir=$timit_dir --save_path=$data_dir

    rm -rf l2_train train_timit

    python Self-Supervised-MDD/scripts/get_model_units.py $data_dir/train/phn_text $data_dir/label_units
    python Self-Supervised-MDD/scripts/get_model_units.py $data_dir/train/trans_g2p $data_dir/trans_units

    # prepare manifest files
    python Self-Supervised-MDD/scripts/generate_manifest.py $all_data_dir\
        --dest $data_dir\
        --segment train\
        --scp_path $data_dir/train/wav.scp
    
    python Self-Supervised-MDD/scripts/generate_manifest.py $all_data_dir\
        --dest $data_dir\
        --segment valid\
        --scp_path $data_dir/dev/wav.scp

    python Self-Supervised-MDD/scripts/generate_manifest.py $all_data_dir\
        --dest $data_dir\
        --segment test\
        --scp_path $data_dir/test/wav.scp

    python Self-Supervised-MDD/scripts/generate_manifest_unlabeled.py $all_data_dir\
        --dest $data_dir\
        --l2_path $l2_arctic_dir

    # prepare labels
    python Self-Supervised-MDD/scripts/generate_labels.py\
        --dest $data_dir\
        --segment train\
        --phn_text_path $data_dir/train/phn_text
    
    python Self-Supervised-MDD/scripts/generate_labels.py\
        --dest $data_dir\
        --segment valid\
        --phn_text_path $data_dir/dev/phn_text

    python Self-Supervised-MDD/scripts/generate_labels.py\
        --dest $data_dir\
        --segment test\
        --phn_text_path $data_dir/test/phn_text

    python Self-Supervised-MDD/scripts/generate_dict.py --data_dir $data_dir

fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Training..."
    
    HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py\
    --config-dir ./Self-Supervised-MDD/config\
    --config-name $config_name\
    common.tensorboard_logdir=$output_dir/$config_name\
    task.data=$data_dir\
    task.trans_path=$trans_path\
    model.w2v_path=$checkpoint_path\
    hydra.run.dir=$output_dir/$config_name\
    checkpoint.save_dir=$output_dir/$config_name

    # task.trans_path=$trans_path\
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Stage 2: Inference..."
    cd fairseq
    python ../Self-Supervised-MDD/scripts/decode.py\
      --checkpoint_path $output_dir/$config_name/checkpoint_best.pt\
      --config_name $output_dir/$config_name/\
      --data_dir $data_dir\
      --segment test\
      --output_dir $output_dir

    cd ..
    #   
fi

if [ $stage -le 3 ]  && [ $stop_stage -ge 3 ]; then
    echo "Step 5: Calculating MDD results..."
    ./Self-Supervised-MDD/scripts/mdd_result.sh $config_name $data_dir test $output_dir|| exit 1;
fi
