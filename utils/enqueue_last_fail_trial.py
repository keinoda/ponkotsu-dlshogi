import optuna
from optuna.trial import TrialState

storage = "sqlite:////volume/param_optimize/model_resnet35x512.db"
study = optuna.load_study(study_name="model_resnet35x512", storage=storage)

# 最後のFAILトライアルを再エンキュー
last_fail = [t for t in study.trials if t.state == TrialState.FAIL][-1]
study.enqueue_trial(last_fail.params)
print(f"Trial {last_fail.number} を再エンキューしました: {last_fail.params}")
