import os
import argparse
from cshogi import CSA

def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="WCSC棋譜をフィルタリングするスクリプト")
    parser.add_argument("software_list_file", help="フィルタリング対象のソフト一覧を記載したテキストファイル (1行1ソフト)")
    parser.add_argument("csa_dir", help="CSAファイルが格納されているディレクトリ")
    args = parser.parse_args()
    return args

def load_software_list(file_path):
    """テキストファイルからソフト一覧を読み込む"""
    # ファイルパスが相対パスの場合は絶対パスに変換
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    
    software_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                software_name = line.strip()
                if software_name and not software_name.startswith('#'):  # 空行とコメント行をスキップ
                    software_list.append(software_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    if not software_list:
        raise ValueError(f"ソフト一覧が空です: {file_path}")
    
    return software_list

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
    print(f"読み込みファイル: {args.software_list_file}")
    print(f"CSAディレクトリ: {args.csa_dir}")
    
    try:
        software_list = load_software_list(args.software_list_file)
        print(f"フィルタリング対象ソフト ({len(software_list)}個): {software_list}")
        
        # CSAファイルのフィルタリングを実行
        print(f"\nCSAファイルのフィルタリングを開始します...")
        filter_csa_files(args.csa_dir, software_list)
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")
        exit(1)
