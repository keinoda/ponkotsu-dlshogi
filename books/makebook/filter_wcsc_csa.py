import os
import requests
from bs4 import BeautifulSoup
import argparse
from cshogi import CSA

def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="WCSC棋譜リンクを抽出して保存するスクリプト")
    parser.add_argument("html_file", help="上位ソフトの順位が記載されたHTMLファイルのURLまたはローカルパス")
    parser.add_argument("csa_dir", help="CSAファイルが格納されているディレクトリ")
    parser.add_argument(
        "--top-n",
        "-n",
        type=int,
        default=10,
        help="抽出対象の上位ソフト数 (デフォルト: 10)",
    )
    args = parser.parse_args()
    if args.top_n <= 0:
        parser.error("--top-n は 1 以上を指定してください")
    return args

def fetch_and_save_html(url, save_path="temp_html.html"):
    """HTMLをダウンロードして一時ファイルに保存"""
    response = requests.get(url)
    response.encoding = "utf-8"  # UTF-8としてエンコーディングを設定
    with open(save_path, 'wb') as f:
        f.write(response.content)
    return save_path

def extract_top_software(html_file, top_n=10):
    """HTMLファイルから上位Nソフトを抽出"""
    # URLの場合は一度ダウンロードして保存
    temp_file = None
    if html_file.startswith("http://") or html_file.startswith("https://"):
        temp_file = fetch_and_save_html(html_file)
        html_file = temp_file
    
    # 直接ファイルを開いてBeautifulSoupで解析
    with open(html_file, 'rb') as f:
        # UTF-8として解析 (標準パーサーを使用)
        soup = BeautifulSoup(f, "html.parser")
    
    # 一時ファイルを削除
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
    
    rows = soup.find_all("tr")[1:]  # ヘッダーを除外
    
    software_data = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > 1:  # 空行をスキップ
            name_cell = cells[1]  # ソフト名が2列目
            rank_cell = cells[-1]  # 順位が最終列
            software_name = name_cell.get_text(strip=True)
            try:
                rank = int(rank_cell.get_text(strip=True))
                software_data.append((rank, software_name))
            except ValueError:
                continue
    
    software_data.sort(key=lambda x: x[0])
    return [name for _, name in software_data[:top_n]]

def filter_csa_files(csa_dir, top_software):
    """指定されたディレクトリ内のCSAファイルをフィルタリング"""
    top_software_set = set(top_software)  # 検索を高速化するためにセットに変換
    parser = CSA.Parser()
    
    total_files = 0
    kept_files = 0
    
    for root, _, files in os.walk(csa_dir):
        for file in files:
            if file.endswith(".csa"):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    # CSAファイルを解析
                    parser.parse_csa_file(file_path)
                    players = parser.names
                    
                    # 両対局者が指定した上位ソフトに含まれるかチェック
                    if len(players) == 2 and players[0] in top_software_set and players[1] in top_software_set:
                        kept_files += 1
                        print(f"保持: {file_path} - {players[0]} vs {players[1]}")
                    else:
                        # 条件を満たさない場合は削除
                        os.remove(file_path)
                        print(f"削除: {file_path} - {players}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"エラー: {file_path} - {str(e)}")
    
    print(f"\n処理結果: 合計{total_files}ファイル中、{kept_files}ファイルを保持、{total_files - kept_files}ファイルを削除しました。")

if __name__ == "__main__":
    args = parse_arguments()
    top_software = extract_top_software(args.html_file, args.top_n)
    print(f"上位{args.top_n}ソフト: {top_software}")
    
    # CSAファイルのフィルタリングを実行
    print(f"\nCSAファイルのフィルタリングを開始します...")
    filter_csa_files(args.csa_dir, top_software)
