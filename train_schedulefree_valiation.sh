echo "resnet10x154_pre_ln_radam_schedulefree_lr_0.0025"
bash train_schedulefree.sh .. resnet10x154_pre_ln_radam_schedulefree_lr_0.0025 ../ShogiAIBookData/ policy_value_network_pre_ln.PolicyValueNetwork  RAdamScheduleFree'()' 0.0025
echo "resnet10x154_pre_ln_radam_schedulefree_lr_1e-4"
bash train_schedulefree.sh .. resnet10x154_pre_ln_radam_schedulefree_lr_1e-4 ../ShogiAIBookData/ policy_value_network_pre_ln.PolicyValueNetwork  RAdamScheduleFree'()' 1e-4
echo "resnet10x154_pre_ln_sgd_schedulefree_lr_1"
bash train_schedulefree.sh .. resnet10x154_pre_ln_sgd_schedulefree_lr_1 ../ShogiAIBookData/ policy_value_network_pre_ln.PolicyValueNetwork  SGDScheduleFree'()' 1
echo "resnet10x154_pre_ln_sgd_schedulefree_lr_1_warmup_100000"
bash train_schedulefree.sh .. resnet10x154_pre_ln_sgd_schedulefree_lr_1_warmup_100000 ../ShogiAIBookData/ policy_value_network_pre_ln.PolicyValueNetwork  SGDScheduleFree'('warmup_steps=100000')' 1
# bash train_schedulefree.sh .. resnet10x154_sgd_schedulefree_lr_1 ../ShogiAIBookData/ policy_value_network_ponkotsu.PolicyValueNetwork  SGDScheduleFree'()' 1