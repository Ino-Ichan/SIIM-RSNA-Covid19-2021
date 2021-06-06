exp=exp001
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# invert white/blackなし
exp=exp002
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# 自分の平均分散で正規化、Invertあり
exp=exp003
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml