# 基于自监督学习的音色无关错误发音检测与诊断

语音自监督预训练-微调的训练范式因为有效利用了海量的无标注数据让模型习得语音表征能力，对 MDD 等多个语音任务带来了效果的提升。但由于自监督预训练的训练目标和 MDD 存在差异，通用模型的表征包含过多信息，而专用模型的表征包含过少信息，直接微调得到的 MDD 模型表现不是最优的。为了解决这个问题，本工作提出模型主干部分选用通用模型Wav2Vec2.0 模型，并在 MDD 的微调阶段加入音色无关的约束进行模型训练的 MDD 方法。该方法能受益于预训练模型对语音的丰富表征能力，但不过拟合到与 MDD 任务无关的说话人音色上。

## 环境设置

```shell
bash ./setup.sh
```

另外需要克隆并编译kaldi，把kaldi项目路径填入path.sh中的local_kaldi_path。

## 数据准备

```shell
python3 ./Self-Supervised-MDD/scripts/timit_downsampling.py --raw_l2_arctic_dir $raw_l2_arctic_dir --output_dir $l2_arctic_dir

Self-Supervised-MDD/scripts/timit_data_prep.sh $timit_dir "60-40" || exit 1;

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
```

以上脚本中的几个参数释义如下：

- timit_dir：TIMIT数据集的路径
- raw_l2_arctic_dir：L2-ARCTIC v5.0数据集的路径
- l2_arctic_dir：脚本输出L2-ARCTIC降采样到16k音频文件的路径
- all_data_dir：timit_dir和l2_arctic_dir所在的路径，它们应该放在同一路径下
- data_dir：脚本输出数据文件的路径

## 模型训练

```shell
# 在TIMIT和L2-Arctic上音色无关地微调Wav2Vec2.0预训练模型，wav2vec_small.pt从https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt下载
HYDRA_FULL_ERROR=1 python -u ./fairseq/fairseq_cli/hydra_train.py\
    --config-dir ./Self-Supervised-MDD/config\
    --config-name finetune_base_vc\
    common.tensorboard_logdir=$output_dir/finetune_base_vc\
    task.data=$data_dir\
    model.w2v_path=wav2vec_small.pt\
    hydra.run.dir=$output_dir/$finetune_base_vc\
    checkpoint.save_dir=$output_dir/$finetune_base_vc
```

## 模型推理

```shell
python ../Self-Supervised-MDD/scripts/decode.py\
      --checkpoint_path $output_dir/finetune_base_vc/checkpoint_best.pt\
      --config_name $output_dir/finetune_base_vc/\
      --data_dir $data_dir\
      --segment test\
      --output_dir $output_dir
```

## MDD结果分析

```shell
. ./path.sh
./Self-Supervised-MDD/scripts/mdd_result.sh finetune_base_vc $data_dir test $output_dir
cat $output_dir/finetune_base_vc/mdd_result
```
