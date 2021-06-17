# exp=exp001
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# exp=exp002
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし
# exp=exp003
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # Focal loss、Invertあり
# exp=exp004
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # Light augmentationのみ、Invertあり
# exp=exp005
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # Light augmentationのみ、Invertあり、CLAHE preprocessing
# exp=exp006
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # Invertなし, softmax
# exp=exp007
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, initial lr 1e-4, from exp003
# exp=exp008
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, initial lr 2e-4, 40 epochs, from exp008
# exp=exp009
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # invert white/blackなし, b3 from exp003
# exp=exp010
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s from exp003
# exp=exp011
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, resnest50d from exp003
# exp=exp012
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, ecaresnet50t from exp003
# exp=exp013
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, eca_nfnet_l1 from exp003
# exp=exp014
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, resnet200d from exp003
# exp=exp015
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp011
# exp=exp016
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, ecaresnet50d_pruned, from exp003
# exp=exp017
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, ecaresnext50t_32x4d, from exp003
# exp=exp018
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, ecaresnet50t from exp013, img_size=640, fine tune
# exp=exp019
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp016, accum=4 , img_size=640, fine tune
# exp=exp020
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# ====================================
# Radar pp
# ====================================

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp020
# exp=exp021
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp021, transform aug false
# exp=exp022
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp021, only light aug 
# exp=exp023
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp021, hard aug -> @30 only light aug 
# exp=exp024
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp021, hard aug -> @20 only light aug 
# exp=exp025
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp021, hard aug adjust epoch
# exp=exp026
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp026, hard aug adjust epoch, crop black
# exp=exp027
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp026, hard aug adjust epoch, image adjust
# exp=exp028
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, tf_efficientnet_b2_ns, freeze bn, from exp026, hard aug adjust epoch, clahe
# exp=exp029
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp026, hard aug adjust epoch
# exp=exp030
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, ecaresnet50t, from exp026, hard aug adjust epoch
# exp=exp031
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp030, hard aug adjust epoch, 5 folds
# exp=exp032
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 m, freeze bn, from exp030, hard aug adjust epoch, 5 folds
# exp=exp033
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp026, hard aug adjust epoch
# exp=exp034
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 1 folds, 384
# exp=exp035
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 1 folds, 320
# exp=exp036
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 1 folds, 256
# exp=exp037
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp034, semi-hard aug(-distort) adjust epoch
# exp=exp038
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp034, middle aug(-distort) adjust epoch
# exp=exp039
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp034, hard -> light adjust epoch
# exp=exp040
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp039, middle aug(-distort) adjust 5 epoch
# exp=exp041
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, hard aug -> middle aug(-distort) adjust 5 epoch, min lr=2e-5
# exp=exp042
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp043, middle aug(-distort) adjust 5 epoch, SAM
# exp=exp043
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, middle aug(-distort) adjust 5 epoch
# exp=exp044
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp042, hard -> light aug adjust cv1 epoch
# exp=exp045
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp042, hard cv1, 2, lr 5e-5 -> 1e-6, few cutout holes
# exp=exp046
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp042, hard cv1, 2, lr 5e-5 -> 1e-6, one cycle lr, few cutout holes
# exp=exp047
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp046, middle aug(-distort), cv1, 2, lr 5e-5 -> 1e-6, few cutout holes
# exp=exp048
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp047, hard cv1, 2, lr 5e-5 -> 1e-6, one cycle lr, few cutout holes
# exp=exp049
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # invert white/blackなし, swin_base_patch4_window12_384, from exp047, hard cv1, 2, lr 5e-5 -> 1e-6, one cycle lr, few cutout holes, bug fix
# exp=exp050
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp046, hard 5 folds, lr 5e-5 -> 1e-6, few cutout holes
# exp=exp051
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, middle aug(-distort) adjust 5 epoch, another split
# exp=exp052
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, middle aug(-distort) adjust 5 epoch, another split
# exp=exp053
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, middle aug(-distort) adjust 5 epoch, weight class
# exp=exp054
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, middle aug(-distort) adjust 5 epoch, freeze bn
# exp=exp055
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp041, middle aug(-distort) adjust 5 epoch, dropout=0.2
# exp=exp056
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# ================================================================================================================================
# Maskを使った実験
# ================================================================================================================================

# # invert white/blackなし, swin_base_patch4_window12_384, from exp039, middle aug(-distort) adjust 5 epoch, with mask pretrain
# exp=exp101
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, swin_base_patch4_window12_384, from exp039, middle aug(-distort) adjust 5 epoch, with mask pretrain -> emb finetune
# exp=exp101_2nd
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# invert white/blackなし, swin_base_patch4_window12_384, from exp039, middle aug(-distort) adjust 5 epoch, with mask pretrain, input change
exp=exp102
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# invert white/blackなし, swin_base_patch4_window12_384, from exp039, middle aug(-distort) adjust 5 epoch, with mask pretrain -> emb finetune
exp=exp102_2nd
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

