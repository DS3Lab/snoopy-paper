for run in 0 1 2 3 4
do
  for noise in 0.0 0.1 0.2 0.3 0.4 0.5
  do
    for dataset in "mnist" "cifar10" "cifar100"
    do
      text="${dataset}_noise_${noise}_run${run}"
      dir="/cluster/scratch/rengglic/output_autokeras/models/${dataset}_noise_${noise}_${run}"
      output="outputs/$dataset/$text.log"

      if [ ! -f "$output" ]; then
        bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_image.py --noise $noise --output $output --dataset $dataset --project_name_suffix $run --max_trials 2
        echo $output
      fi
    done

    dataset="imdb_reviews"
    text="${dataset}_noise_${noise}_run${run}"
    output="outputs/$dataset/$text.log"
    dir="/cluster/scratch/rengglic/output_autokeras/models/${dataset}_noise_${noise}_${run}"
    if [ ! -f "$output" ]; then
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfds.py --noise $noise --output $output --dataset $dataset --project_name_suffix $run --max_trials 5 && rm -rf $dir
      echo $output
    fi

    dataset="sst2"
    text="${dataset}_noise_${noise}_run${run}"
    output="outputs/$dataset/$text.log"
    dir="/cluster/scratch/rengglic/output_autokeras/models/${dataset}_noise_${noise}_${run}"
    if [ ! -f "$output" ]; then
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_tfds.py --noise $noise --output $output --dataset $dataset --test_split validation --feature sentence --project_name_suffix $run --max_trials 4 && rm -rf $dir
      echo $output
    fi

    dataset="yelp"
    text="${dataset}_noise_${noise}_run${run}"
    output="outputs/$dataset/$text.log"
    dir="/cluster/scratch/rengglic/output_autokeras/models/${dataset}_noise_${noise}_${run}"
    if [ ! -f "$output" ]; then
      bsub -n 4 -W 24:00 -R "rusage[mem=11000,ngpus_excl_p=1]" -o outputs/$dataset/lsf.$text python main_text_csv.py --noise $noise --output $output --project_name_suffix $run --max_trials 2 && rm -rf $dir
      echo $output
    fi
  done
done
