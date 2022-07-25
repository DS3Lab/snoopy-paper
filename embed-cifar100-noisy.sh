for dataset in cifar100-noisy
do
  for i in `seq 0 19`
  do
    python embed.py $dataset $i
  done
done
