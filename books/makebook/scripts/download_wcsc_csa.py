import argparse
import os
import re
from urllib.parse import unquote, urlparse

import requests

DEFAULT_LIST_URL = "https://www.computer-shogi.org/live/wcsc36/list.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="WCSC list.txt から CSA 棋譜を一括ダウンロードします"
    )
    parser.add_argument(
        "save_dir",
        help="保存先ディレクトリ",
    )
    parser.add_argument(
        "--list-url",
        default=DEFAULT_LIST_URL,
        help=f"list.txt のURL (デフォルト: {DEFAULT_LIST_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTPタイムアウト秒 (デフォルト: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルがあっても上書きする",
    )
    return parser.parse_args()


def fetch_list_text(list_url, timeout):
    response = requests.get(list_url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def extract_csa_urls(text):
    urls = re.findall(r"https?://[^\s\"'<>]+\.csa", text)
    # 順序を維持したまま重複排除
    return list(dict.fromkeys(urls))


def filename_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path)
    return unquote(filename)


def download_file(url, save_path, timeout):
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"list.txt を取得: {args.list_url}")
    list_text = fetch_list_text(args.list_url, args.timeout)

    urls = extract_csa_urls(list_text)
    if not urls:
        print(".csa URL が見つかりませんでした。")
        return

    print(f"検出URL数: {len(urls)}")

    downloaded = 0
    skipped = 0
    failed = 0

    for index, url in enumerate(urls, start=1):
        filename = filename_from_url(url)
        save_path = os.path.join(args.save_dir, filename)

        if os.path.exists(save_path) and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(urls)}] スキップ: {filename}")
            continue

        try:
            download_file(url, save_path, args.timeout)
            downloaded += 1
            print(f"[{index}/{len(urls)}] 保存: {filename}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(urls)}] 失敗: {url} ({exc})")

    print("\n完了")
    print(f"保存: {downloaded}")
    print(f"スキップ: {skipped}")
    print(f"失敗: {failed}")


if __name__ == "__main__":
    main()
