for dataset in cifar10-random2 
do
  for i in `seq 0 19`
  do
    python embed.py $dataset $i
  done
done
