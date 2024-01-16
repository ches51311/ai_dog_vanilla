# should converage about 1500 iter, 8min with rtx 4080 (press F on UI to faster)
python3 main.py --net_type=linear --reward_type=simple --times=5000 --use_critic
# should converage about 2500 iter
python3 main.py --net_type=linear --reward_type=HP_MP --times=5000

# should converage about 1000 iter 
time python3 main.py --net_type=linear_recall --reward_type=sai --times=5000
# should converage about 2500 iter 
time python3 main.py --net_type=linear_recall --reward_type=simple --times=5000
# half success
time python3 main.py --net_type=linear_recall --reward_type=HP_MP --times=10000

# half success
time python3 main.py --net_type=transformer_recall --reward_type=sai --times=5000
# success
time python3 main.py --net_type=transformer_recall --reward_type=simple --times=5000
# bad
time python3 main.py --net_type=transformer_recall --reward_type=HP_MP --times=5000 --use_critic

