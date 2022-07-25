for dataset in cifar10-worst 
do
  for i in `seq 0 19`
  do
    python embed.py $dataset $i
  done
done
