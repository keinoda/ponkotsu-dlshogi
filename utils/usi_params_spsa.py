"""
SPSA (Simultaneous Perturbation Stochastic Approximation) による
USIエンジンのパラメータ最適化

YaneuraOu tune / Stockfish fishtest 方式をベースに、
バッチ評価（N局/評価 × M イテレーション）向けに調整。

YaneuraOu .params方式との対応:
  - c_end = (max - min) / 20   → 正規化空間で 0.05 (5%)
  - r_end = 0.002 (fishtest per-game更新向け)
    → バッチ評価向けにスケーリング: r_end_norm (デフォルト 0.01)
  - 線形減衰: c_0 = 2*c_end → c_end, r_0 = 2*r_end → r_end (YaneuraOu方式)
"""

import argparse
import json
import logging
import os
import sys

import cshogi.cli

logger = logging.getLogger(__name__)


def parse_params(s):
    """'key:value,key:value,...' 形式のパラメータ文字列をdictに変換"""
    params = {}
    for kv in s.split(','):
        k, v = kv.split(':')
        params[k] = int(v)
    return params


def parse_ranges(s):
    """'key:min~max,key:min~max,...' 形式の範囲文字列をdictに変換"""
    ranges = {}
    for kv in s.split(','):
        k, v = kv.split(':')
        lo, hi = v.split('~')
        ranges[k] = (int(lo), int(hi))
    return ranges


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def run_match(command1, command2, options1, options2, params, args, params2=None):
    """パラメータを設定してエンジン対局を実行し、先後別の勝率を含む結果を返す。

    cshogi.cliは先後交互対局(n%2==0: engine1=先手, n%2==1: engine1=後手)。
    callbackの累積値の差分から1局ごとの勝敗を追跡し先後別に集計する。

    Args:
        params: engine1に適用するパラメータdict
        params2: engine2に適用するパラメータdict (dev-vs-dev用、Noneならoptions2のみ)

    Returns:
        dict: {'total': 総合勝率, 'black': engine1先手時の勝率, 'white': engine1後手時の勝率}
    """
    opts1 = dict(options1)
    for k, v in params.items():
        opts1[k] = v
    opts2 = dict(options2)
    if params2 is not None:
        for k, v in params2.items():
            opts2[k] = v

    class Callback:
        def __init__(self):
            self.prev_e1_won = 0
            self.prev_draw = 0
            self.prev_total = 0
            self.black_wins = 0
            self.black_draws = 0
            self.black_games = 0
            self.white_wins = 0
            self.white_draws = 0
            self.white_games = 0
            self.total_wr = 0.0

        def __call__(self, result):
            e1 = result['engine1_won']
            dr = result['draw']
            total = result['total']

            d_e1 = e1 - self.prev_e1_won
            d_dr = dr - self.prev_draw
            game_n = self.prev_total
            engine1_is_black = (game_n % 2 == 0)

            if engine1_is_black:
                self.black_games += 1
                if d_e1: self.black_wins += 1
                if d_dr: self.black_draws += 1
            else:
                self.white_games += 1
                if d_e1: self.white_wins += 1
                if d_dr: self.white_draws += 1

            self.prev_e1_won = e1
            self.prev_draw = dr
            self.prev_total = total
            self.total_wr = (e1 + dr / 2) / total
            return True

    callback = Callback()
    kwargs = dict(
        engine1=command1,
        engine2=command2,
        options1=opts1,
        options2=opts2,
        names=[args.name, None],
        games=args.games,
        mate_win=True,
        byoyomi=args.byoyomi,
        draw=args.max_turn,
        opening=args.opening,
        opening_moves=args.opening_moves,
        keep_process=True,
        is_display=False,
        debug=args.debug,
        callback=callback,
    )
    if args.csa is not None:
        kwargs['csa'] = args.csa

    cshogi.cli.main(**kwargs)

    black_wr = (callback.black_wins + callback.black_draws / 2) / callback.black_games if callback.black_games > 0 else 0.5
    white_wr = (callback.white_wins + callback.white_draws / 2) / callback.white_games if callback.white_games > 0 else 0.5

    return {'total': callback.total_wr, 'black': black_wr, 'white': white_wr}


def spsa_optimize(args):
    """
    SPSA最適化メインループ

    正規化空間 [0,1] で計算 (YaneuraOu方式の線形減衰):

      c_end_norm = 0.05 (= range/20 / range, YaneuraOu .params の step に相当)
      r_end_norm = 学習率 (YaneuraOu の delta=0.002 のバッチ評価版)

      線形減衰 (t: 1→0):
        c_k = c_end_norm * (1 + t)   ... 0.10 → 0.05
        r_k = r_end_norm * (1 + t)   ... 2*r_end → r_end

    YaneuraOu .params との関係:
      c_end(実値) = range * c_end_norm = range/20  ← 一致
      r_end: fishtest per-game方式では 0.002、
             バッチ50局方式では √50 ≈ 7倍低ノイズなので 0.01 程度が妥当
    """
    import numpy as np

    # パラメータの初期値と範囲
    theta = parse_params(args.init_params)
    ranges = parse_ranges(args.suggest_params)
    param_names = list(theta.keys())

    # エンジンオプション
    options1, options2 = {}, {}
    for i, kvs_str in enumerate([args.options1, args.options2]):
        opts = [options1, options2][i]
        if kvs_str:
            for kv_str in kvs_str.split(','):
                k, v = kv_str.split(':', 1)
                opts[k] = v

    # SPSA 係数 (正規化空間)
    c_end = args.spsa_c_end   # YaneuraOu: step = range/20 → 正規化で 0.05
    r_end = args.spsa_r_end   # 学習率
    N = args.spsa_iterations

    # ログ/チェックポイント
    log_path = args.spsa_log
    checkpoint_path = args.spsa_checkpoint
    start_iter = 0

    # 正規化空間 phi ∈ [0,1] に変換
    phi = {}
    for name in param_names:
        lo, hi = ranges[name]
        phi[name] = (float(theta[name]) - lo) / (hi - lo)

    # チェックポイントから復元
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            cp = json.load(f)
        start_iter = cp['iteration'] + 1
        if 'phi' in cp:
            phi = {k: float(v) for k, v in cp['phi'].items()}
        else:
            for name in param_names:
                lo, hi = ranges[name]
                phi[name] = (float(cp['theta'][name]) - lo) / (hi - lo)
        logger.info(f"Resumed from checkpoint: iteration {start_iter}")

    # 最適化対象の手番
    optimize_side = args.optimize_side  # 'total', 'black', 'white'

    logger.info("SPSA optimization start (YaneuraOu linear decay, normalized space)")
    logger.info(f"  Initial params: {theta}")
    logger.info(f"  Ranges: {ranges}")
    logger.info(f"  Iterations: {N}, Games/eval: {args.games}")
    logger.info(f"  Optimize side: {optimize_side}")
    logger.info(f"  Dev-vs-dev: {args.spsa_dev_vs_dev}")
    logger.info(f"  c_end={c_end} (perturbation, YaneuraOu step/range=1/20={1/20:.3f})")
    logger.info(f"  r_end={r_end} (learning rate)")
    for name in param_names:
        lo, hi = ranges[name]
        logger.info(f"  {name}: range=[{lo},{hi}], c_end(real)={c_end*(hi-lo):.1f}")

    for k in range(start_iter, N):
        # 線形減衰 t: 1→0 (YaneuraOu方式)
        remaining = N - 1 - k
        t = remaining / max(N - 1, 1)

        c_k = c_end * (1.0 + t)  # 2*c_end → c_end
        r_k = r_end * (1.0 + t)  # 2*r_end → r_end

        # Bernoulli ±1 の摂動ベクトル
        delta = {name: np.random.choice([-1, 1]) for name in param_names}

        # 正規化空間で摂動 → 実パラメータに変換
        params_plus = {}
        params_minus = {}
        for name in param_names:
            lo, hi = ranges[name]
            r = hi - lo
            phi_p = clamp(phi[name] + c_k * delta[name], 0.0, 1.0)
            phi_m = clamp(phi[name] - c_k * delta[name], 0.0, 1.0)
            params_plus[name]  = clamp(int(round(lo + phi_p * r)), lo, hi)
            params_minus[name] = clamp(int(round(lo + phi_m * r)), lo, hi)

        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {k}/{N-1}  t={t:.3f}  c_k={c_k:.4f}  r_k={r_k:.4f}")
        logger.info(f"  theta+ = {params_plus}")
        logger.info(f"  theta- = {params_minus}")

        # θ+ で対局
        logger.info(f"  Playing theta+ match ({args.games} games)...")
        result_plus = run_match(
            args.command1, args.command2, options1, options2, params_plus, args
        )

        # θ- で対局
        logger.info(f"  Playing theta- match ({args.games} games)...")
        result_minus = run_match(
            args.command1, args.command2, options1, options2, params_minus, args
        )

        # 最適化対象の勝率を選択
        win_rate_plus = result_plus[optimize_side]
        win_rate_minus = result_minus[optimize_side]

        logger.info(f"  [{optimize_side}] win_rate+={win_rate_plus:.3f}, win_rate-={win_rate_minus:.3f}")
        logger.info(f"    plus:  total={result_plus['total']:.3f} black={result_plus['black']:.3f} white={result_plus['white']:.3f}")
        logger.info(f"    minus: total={result_minus['total']:.3f} black={result_minus['black']:.3f} white={result_minus['white']:.3f}")

        # dev-vs-dev: θ+ vs θ- の直接対戦 (Fishtest方式)
        result_pm = None
        win_rate_pm = None
        if args.spsa_dev_vs_dev:
            logger.info(f"  Playing theta+ vs theta- match ({args.games} games)...")
            result_pm = run_match(
                args.command1, args.command1, options1, options1,
                params_plus, args, params2=params_minus
            )
            win_rate_pm = result_pm[optimize_side]
            logger.info(f"  [{optimize_side}] win_rate(+vs-)={win_rate_pm:.3f}")
            logger.info(f"    +vs-:  total={result_pm['total']:.3f} black={result_pm['black']:.3f} white={result_pm['white']:.3f}")

        # 正規化空間で勾配推定と更新
        # dev-vs-dev有効時: score_diff = (WR+ - WR-) + (WR_pm - 0.5)
        #   WR_pm - 0.5 はθ+がθ-に対する超過勝率 (0なら情報なし)
        score_diff = win_rate_plus - win_rate_minus
        if args.spsa_dev_vs_dev and win_rate_pm is not None:
            score_diff += (win_rate_pm - 0.5)
        for name in param_names:
            g_hat = score_diff / (2.0 * c_k * delta[name])
            phi[name] += r_k * g_hat
            phi[name] = clamp(phi[name], 0.0, 1.0)

        # 実パラメータに変換
        theta = {}
        for name in param_names:
            lo, hi = ranges[name]
            theta[name] = clamp(int(round(lo + phi[name] * (hi - lo))), lo, hi)

        logger.info(f"  Updated theta = {theta}")

        # ログ出力
        if log_path:
            with open(log_path, 'a') as f:
                log_entry = {
                    'iteration': k,
                    'theta': theta,
                    'phi': {n: round(phi[n], 6) for n in param_names},
                    'theta_plus': params_plus,
                    'theta_minus': params_minus,
                    'optimize_side': optimize_side,
                    'win_rate_plus': win_rate_plus,
                    'win_rate_minus': win_rate_minus,
                    'result_plus': result_plus,
                    'result_minus': result_minus,
                    'result_pm': result_pm,
                    'win_rate_pm': win_rate_pm,
                    'score_diff': score_diff,
                    'c_k': c_k,
                    'r_k': r_k,
                }
                f.write(json.dumps(log_entry) + '\n')

        # チェックポイント保存
        if checkpoint_path:
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'iteration': k,
                    'theta': theta,
                    'phi': {n: phi[n] for n in param_names},
                }, f)

    logger.info(f"\nOptimization finished. Final params: {theta}")

    # 最終パラメータで検証対局
    if args.spsa_verify_games > 0:
        logger.info(f"\nRunning verification match ({args.spsa_verify_games} games)...")
        old_games = args.games
        args.games = args.spsa_verify_games
        result = run_match(
            args.command1, args.command2, options1, options2, theta, args
        )
        args.games = old_games
        logger.info(f"Final verification: total={result['total']:.3f} black={result['black']:.3f} white={result['white']:.3f}")

    return theta


def main():
    parser = argparse.ArgumentParser(description='SPSA parameter optimizer for USI engines (YaneuraOu linear decay)')
    parser.add_argument('command1', help='Engine 1 (optimized)')
    parser.add_argument('command2', help='Engine 2 (baseline)')
    parser.add_argument('--options1', default='')
    parser.add_argument('--options2', default='')
    parser.add_argument('--name', default=None)
    parser.add_argument('--games', type=int, default=50, help='Games per evaluation (default: 50)')
    parser.add_argument('--byoyomi', type=int, default=2000)
    parser.add_argument('--max_turn', type=int, default=320)
    parser.add_argument('--opening', default=None)
    parser.add_argument('--opening_moves', type=int, default=24)
    parser.add_argument('--csa', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--init_params',
                        default='C_init:123,C_base:32303,C_fpu_reduction:14,C_init_root:155,C_base_root:26661,Softmax_Temperature:173',
                        help='Initial parameters (default: TPE top-5 median)')
    parser.add_argument('--suggest_params',
                        default='C_init:100~200,C_base:20000~50000,C_fpu_reduction:0~40,C_init_root:100~200,C_base_root:20000~50000,Softmax_Temperature:100~200',
                        help='Parameter ranges')

    # SPSA hyperparameters (YaneuraOu linear decay, normalized space)
    parser.add_argument('--spsa_iterations', type=int, default=50, help='Number of SPSA iterations')
    parser.add_argument('--spsa_c_end', type=float, default=0.05,
                        help='C_End in normalized [0,1] space (default: 0.05 = range/20, matches YaneuraOu)')
    parser.add_argument('--spsa_r_end', type=float, default=0.01,
                        help='R_End learning rate in normalized space (default: 0.01; YaneuraOu uses 0.002 for per-game updates)')
    parser.add_argument('--optimize_side', choices=['total', 'black', 'white'], default='total',
                        help='Which win rate to optimize: total (default), black (engine1 as sente), white (engine1 as gote)')
    parser.add_argument('--spsa_dev_vs_dev', action='store_true', default=False,
                        help='Enable dev-vs-dev match (theta+ vs theta-) per iteration for better gradient estimation (Fishtest style). Adds 50%% more games per iteration.')

    parser.add_argument('--spsa_log', type=str, default=None, help='SPSA log file (JSONL)')
    parser.add_argument('--spsa_checkpoint', type=str, default=None, help='Checkpoint file for resume')
    parser.add_argument('--spsa_verify_games', type=int, default=200, help='Final verification games (0 to skip)')

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    theta = spsa_optimize(args)
    print(f"\nFinal optimized parameters:")
    for k, v in theta.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
