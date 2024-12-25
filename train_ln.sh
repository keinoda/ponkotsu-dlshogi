trap "kill 0" SIGINT

bash train_post_ln.sh .. ../ShogiAIBookData/ & bash train_pre_ln.sh .. ../ShogiAIBookData/

wait