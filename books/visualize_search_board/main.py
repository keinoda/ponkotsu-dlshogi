import pickle
import argparse
import cshogi
import cshogi.KIF
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# グローバル変数として指し手のリストを保存
moves_list = []
pickle_file_path = ""
last_modified_time = 0

class PickleFileHandler(FileSystemEventHandler):
    """Pickleファイルの変更を監視するハンドラー"""
    
    def on_modified(self, event):
        global last_modified_time, pickle_file_path
        if event.is_directory:
            return
        
        if event.src_path == pickle_file_path:
            # ファイルの更新時刻をチェックして重複読み込みを防ぐ
            current_modified_time = os.path.getmtime(pickle_file_path)
            if current_modified_time > last_modified_time:
                print(f"Pickle file updated: {pickle_file_path}")
                load_pickle_file()

def load_pickle_file():
    """Pickleファイルを読み込む"""
    global moves_list, last_modified_time, pickle_file_path
    try:
        with open(pickle_file_path, "rb") as f:
            new_moves_list = pickle.load(f)
        
        # 成功した場合のみ更新
        moves_list = new_moves_list
        last_modified_time = os.path.getmtime(pickle_file_path)
        print(f"Successfully loaded {len(moves_list)} board sequences")
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        # エラーの場合は既存のデータを保持

def setup_file_watcher():
    """ファイル監視を設定"""
    global pickle_file_path
    if not os.path.exists(pickle_file_path):
        print(f"Warning: Pickle file does not exist: {pickle_file_path}")
        return None
    
    event_handler = PickleFileHandler()
    observer = Observer()
    
    # ファイルのディレクトリを監視
    watch_directory = os.path.dirname(os.path.abspath(pickle_file_path))
    observer.schedule(event_handler, watch_directory, recursive=False)
    
    observer.start()
    print(f"Started watching for changes in: {watch_directory}")
    return observer

def get_move_display(move, move_index):
    """指し手を表示用の文字列に変換"""
    if isinstance(move, int):
        try:
            # KIF形式に変換
            move_str = cshogi.KIF.move_to_kif(move)
            return f"{move_index + 1}.{move_str}"
        except:
            return f"{move_index + 1}.{move}"
    else:
        return f"{move_index + 1}.{move}"

def make_board_at_move(history, move_index):
    """指定した手番までの盤面を作成（メモリ効率化）"""
    board = cshogi.Board()
    
    # move_index が -1 の場合は初期局面を返す
    if move_index == -1:
        return board
    
    for i, move in enumerate(history):
        if i > move_index:
            break
        try:
            if isinstance(move, int):
                board.push(move)
            else:
                board.push_usi(move)
        except Exception as e:
            print(f"Error pushing move {move}: {e}")
            continue
    return board

@app.route('/')
def index():
    """メインページ：指し手の一覧を表示（軽量化）"""
    # 初期表示は最初の10盤面のみ
    initial_limit = 10
    total_boards = len(moves_list)
    display_boards = min(initial_limit, total_boards)
    
    boards_data = []
    for i in range(display_boards):
        history = moves_list[i]
        if len(history) > 0:
            # 初期局面の盤面を生成
            initial_board = cshogi.Board()
            board_svg = initial_board.to_svg()
            
            # 全ての手順を表示用として保存
            move_preview = []
            for j, move in enumerate(history):
                move_preview.append(get_move_display(move, j))
            
            boards_data.append({
                'index': i,
                'board_index': i + 1,
                'board_svg': board_svg,
                'sfen': initial_board.sfen(),
                'move_count': len(history),
                'move_preview': move_preview
            })
    
    return render_template('index.html', 
                         boards=boards_data, 
                         has_more_boards=total_boards > display_boards,
                         total_boards=total_boards,
                         loaded_boards=display_boards)

@app.route('/api/board/<int:board_index>/<move_index>')
def get_board_state(board_index, move_index):
    """APIエンドポイント：指定した手番の盤面データを返す"""
    if board_index < 0 or board_index >= len(moves_list):
        return jsonify({'error': 'Board not found'}), 404
    
    # move_indexを整数に変換
    try:
        move_index = int(move_index)
    except ValueError:
        return jsonify({'error': 'Invalid move index format'}), 400
    
    history = moves_list[board_index]
    
    if move_index < -1 or move_index >= len(history):
        return jsonify({'error': 'Invalid move index'}), 400
    
    # 指定した手番までの盤面を作成
    board = cshogi.Board()
    
    # move_index が -1 の場合は初期局面を返す
    if move_index != -1:
        for i in range(move_index + 1):
            if i >= len(history):
                break
            try:
                move = history[i]
                if isinstance(move, int):
                    board.push(move)
                else:
                    board.push_usi(move)
            except Exception as e:
                print(f"Error pushing move {move}: {e}")
                continue
    
    sfen = board.sfen()
    svg = board.to_svg()
    
    # 現在の手の表示
    current_move = "初期局面" if move_index == -1 else get_move_display(history[move_index], move_index)
    
    return jsonify({
        'board_svg': svg,
        'sfen': sfen,
        'current_move': current_move,
        'move_index': move_index,
        'total_moves': len(history),
        'has_prev': move_index > -1,
        'has_next': move_index < len(history) - 1
    })

@app.route('/api/boards')
def get_boards_paginated():
    """盤面データをページネーションで取得するAPIエンドポイント"""
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 10))
    
    # 範囲チェック
    total_boards = len(moves_list)
    if offset >= total_boards:
        return jsonify({'boards': [], 'has_more': False, 'total': total_boards})
    
    end_index = min(offset + limit, total_boards)
    boards_data = []
    
    for i in range(offset, end_index):
        history = moves_list[i]
        if len(history) > 0:
            initial_board = cshogi.Board()
            boards_data.append({
                'index': i,
                'board_index': i + 1,
                'board_svg': initial_board.to_svg(),
                'sfen': initial_board.sfen(),
                'move_count': len(history),
                'move_preview': [get_move_display(mv, j)
                               for j, mv in enumerate(history[:10])],
                'has_more_moves': len(history) > 10
            })
    
    return jsonify({
        'boards': boards_data,
        'has_more': end_index < total_boards,
        'total': total_boards,
        'loaded': end_index
    })

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('pickle_file')
    args.add_argument('--port', type=int, default=5000)
    args = args.parse_args()
    
    # グローバル変数を設定
    pickle_file_path = os.path.abspath(args.pickle_file)
    
    # 初回読み込み
    print(f"Initial loading of pickle file: {pickle_file_path}")
    load_pickle_file()
    
    # ファイル監視を開始
    observer = setup_file_watcher()
    
    try:
        print(f"Starting web server on http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        if observer:
            observer.stop()
            observer.join()
            print("File watcher stopped")
