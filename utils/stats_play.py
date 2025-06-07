import re
import argparse
from collections import defaultdict

def calculate_win_rates(log_text):
    # プレイヤーごとの成績を追跡する辞書
    sente_stats = defaultdict(lambda: [0, 0, 0])  # [勝, 負, 引分]
    gote_stats = defaultdict(lambda: [0, 0, 0])   # [勝, 負, 引分]

    # 対戦カードごとの成績を追跡する辞書
    matchups = defaultdict(lambda: defaultdict(lambda: [[0, 0, 0], [0, 0, 0]]))  # [先手時[勝,負,引], 後手時[勝,負,引]]

    # パターンのコンパイル
    start_pattern = re.compile(r'([^ ]+) vs ([^ ]+) start\.')
    result_pattern = re.compile(r'まで\d+手で(先手|後手)の勝ち')
    draw_pattern = re.compile(r'まで\d+手で千日手')

    sente = None
    gote = None

    # 行ごとに処理
    lines = log_text.strip().split('\n')
    for line in lines:
        # 対局開始行の処理
        start_match = start_pattern.search(line)
        if start_match:
            sente = start_match.group(1)
            gote = start_match.group(2)
            continue

        # 結果行の処理（勝敗）
        result_match = result_pattern.search(line)
        if result_match and sente is not None and gote is not None:
            winner = result_match.group(1)
            if winner == "先手":
                sente_stats[sente][0] += 1  # 先手の勝ち
                gote_stats[gote][1] += 1    # 後手の負け

                # 対戦カードの成績も更新
                matchups[sente][gote][0][0] += 1  # sente が先手で勝ち
                matchups[gote][sente][1][1] += 1  # gote が後手で負け
            else:  # 後手の勝ち
                sente_stats[sente][1] += 1  # 先手の負け
                gote_stats[gote][0] += 1    # 後手の勝ち

                # 対戦カードの成績も更新
                matchups[sente][gote][0][1] += 1  # sente が先手で負け
                matchups[gote][sente][1][0] += 1  # gote が後手で勝ち
            continue

        # 千日手の処理（引き分け）
        draw_match = draw_pattern.search(line)
        if draw_match and sente is not None and gote is not None:
            sente_stats[sente][2] += 1  # 先手の引き分け
            gote_stats[gote][2] += 1    # 後手の引き分け

            # 対戦カードの成績も更新
            matchups[sente][gote][0][2] += 1  # sente が先手で引分
            matchups[gote][sente][1][2] += 1  # gote が後手で引分

    # 結果を出力
    # すべてのプレイヤーをリストで取得してソート
    all_players = sorted(list(set(sente_stats.keys()) | set(gote_stats.keys())))

    # 1. 各プレイヤーの先手・後手の総合成績
    print("== 各プレイヤーの先手・後手の総合成績 ==")
    for player in all_players:
        # 先手（Black）としての成績
        black_wins, black_losses, black_draws = sente_stats[player]
        black_total = black_wins + black_draws + black_losses

        # 引き分けを0.5勝として計算
        black_win_points = black_wins + 0.5 * black_draws
        black_win_rate = black_win_points / black_total * 100 if black_total > 0 else 0

        print(f"{player} playing Black: {black_wins}-{black_losses}-{black_draws} ({black_win_rate:.1f}%)")

        # 後手（White）としての成績
        white_wins, white_losses, white_draws = gote_stats[player]
        white_total = white_wins + white_draws + white_losses

        # 引き分けを0.5勝として計算
        white_win_points = white_wins + 0.5 * white_draws
        white_win_rate = white_win_points / white_total * 100 if white_total > 0 else 0

        print(f"{player} playing White: {white_wins}-{white_losses}-{white_draws} ({white_win_rate:.1f}%)")
        print()

    # 2. 各対戦カードごとの成績
    print("\n== 対戦カードごとの成績 ==")
    for player1 in all_players:
        for player2 in all_players:
            if player1 >= player2:  # 重複を避けるため、同じ対戦は1回だけ表示
                continue

            # player1 が先手の成績
            p1_black_stats = matchups[player1][player2][0]
            p1_black_wins, p1_black_losses, p1_black_draws = p1_black_stats
            p1_black_total = sum(p1_black_stats)
            if p1_black_total > 0:
                p1_black_win_points = p1_black_wins + 0.5 * p1_black_draws
                p1_black_win_rate = p1_black_win_points / p1_black_total * 100
                print(f"{player1} playing Black vs {player2}: {p1_black_wins}-{p1_black_losses}-{p1_black_draws} ({p1_black_win_rate:.1f}%)")

            # player1 が後手の成績
            p1_white_stats = matchups[player1][player2][1]
            p1_white_wins, p1_white_losses, p1_white_draws = p1_white_stats
            p1_white_total = sum(p1_white_stats)
            if p1_white_total > 0:
                p1_white_win_points = p1_white_wins + 0.5 * p1_white_draws
                p1_white_win_rate = p1_white_win_points / p1_white_total * 100
                print(f"{player1} playing White vs {player2}: {p1_white_wins}-{p1_white_losses}-{p1_white_draws} ({p1_white_win_rate:.1f}%)")

            # 対戦カードの総合成績
            total_wins = p1_black_wins + p1_white_wins
            total_losses = p1_black_losses + p1_white_losses
            total_draws = p1_black_draws + p1_white_draws
            total_games = p1_black_total + p1_white_total

            if total_games > 0:
                total_win_points = total_wins + 0.5 * total_draws
                total_win_rate = total_win_points / total_games * 100
                print(f"{player1} total vs {player2}: {total_wins}-{total_losses}-{total_draws} ({total_win_rate:.1f}%)")
                print()

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='将棋の対局ログから先後ごとの勝率を計算します')
    parser.add_argument('log_file', help='対局ログファイルのパス')
    args = parser.parse_args()

    # ファイルから読み込む
    with open(args.log_file, 'r') as f:
        log_text = f.read()

    calculate_win_rates(log_text)
