# should converage about 1500 iter, 8min with rtx 4080 (press F on UI to faster)
python3 main.py --net_type=linear --reward_type=simple --times=5000
# should converage about 2500 iter
python3 main.py --net_type=linear --reward_type=HP_MP --times=5000