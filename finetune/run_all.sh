for run in 0 1 2 3 4
do
  for noise in 0.0 0.1 0.2 0.3 0.4 0.5
  do
    lr=0.01
    reg=0.0
    for dataset in "cifar10" "cifar100"
    do
      text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"

      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_image.py --lr $lr --noise $noise --reg $reg --output outputs/$dataset/$text.log --dataset $dataset
    done

    lr=0.00002
    reg=0.0

    dataset="imdb_reviews"
    text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"
    bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfds.py --lr $lr --noise $noise --output outputs/$dataset/$text.log --dataset $dataset

    dataset="sst2"
    text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"
    bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfds.py --lr $lr --noise $noise --output outputs/$dataset/$text.log --dataset $dataset --test_split validation --feature sentence

    dataset="yelp"
    text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"
    bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfrecords.py --lr $lr --noise $noise --output outputs/$dataset/$text.log
  done
done

for run in 0 1 2 3 4
do
  for lr in 0.00001 0.00005
  do
    for noise in 0.0 0.1 0.2 0.3 0.4 0.5
    do
      dataset="imdb_reviews"
      text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfds.py --lr $lr --noise $noise --output outputs/$dataset/$text.log --dataset $dataset

      dataset="sst2"
      text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfds.py --lr $lr --noise $noise --output outputs/$dataset/$text.log --dataset $dataset --test_split validation --feature sentence

      dataset="yelp"
      text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfrecords.py --lr $lr --noise $noise --output outputs/$dataset/$text.log
    done
  done
done

for run in 0 1 2 3 4
do
  for lr in 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1
  do
    for reg in 0.000001 0.000003 0.00001 0.00003 0.0001 0.0003 0.001 0.0
    do
    for noise in 0.0 0.1 0.2 0.3 0.4 0.5
      do
        for dataset in "cifar10" "cifar100"
        do
          text="${dataset}_noise_${noise}_lr_${lr}_reg_${reg}_run${run}"

          bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_image.py --lr $lr --noise $noise --reg $reg --output outputs/$dataset/$text.log --dataset $dataset
        done
      done
    done
  done
done
