# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 1 folds, 384
# exp=exp035
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 5 folds, 384, bs=32*2
# exp=exp201
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp201, mid aug(-distort) aug adjust epoch, 5 folds, 384, bs=32*2
# exp=exp202
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp201, hard aug adjust epoch, 5 folds, 384, bs=32*2, few cutout holes
# exp=exp203
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp030, hard aug adjust epoch
# exp=exp204
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 l, freeze bn, from exp030, middle aug(-distort) adjust epoch, bs=8*8
# exp=exp205
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# invert white/blackなし, b7, freeze bn, from exp030, middle aug(-distort) adjust epoch, bs=8*8
exp=exp206
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml