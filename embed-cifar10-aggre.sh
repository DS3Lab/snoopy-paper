for dataset in cifar10-aggre 
do
  for i in `seq 0 19`
  do
    python embed.py $dataset $i
  done
done
