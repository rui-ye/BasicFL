START_TIME=`date +%s`
data_partition="noniid"
# data_partition="noniid-skew-2"
iternum=200
client=10
beta=0.1
dataset="cifar10"
model="resnet20_cifar"
sample_fraction=1.0
load_path=None
dir_path=./logs/${dataset}_${data_partition}_${model}_beta${beta}_it${iternum}_c${client}_p${sample_fraction}
mkdir $dir_path

# ===================== Run in the current session / Results printed in the screen. =====================
python main.py --alg $alg --dataset $dataset --gpu "1" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta


# # ===================== Will continue running even if the session is disconnected / Results saved in the logs. =====================
# alg="fedavg"
# nohup python -u main.py --alg $alg --dataset $dataset --gpu "1" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/${alg}_${START_TIME}.log &

# alg="fedprox"
# mu=0.01
# nohup python -u main.py --alg $alg --mu $mu --dataset $dataset --gpu "1" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/${alg}_${mu}_${START_TIME}.log &

# alg="scaffold"
# nohup python -u main.py --alg $alg --dataset $dataset --gpu "1" --partition $data_partition --model $model --n_client $client --sample_fraction $sample_fraction --n_iteration $iternum --beta $beta > $dir_path/${alg}_${START_TIME}.log &
