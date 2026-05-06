# SPSA定期ノート設定手順

## セットアップ

```bash
# ユーザーunitディレクトリにシンボリックリンクを作成
mkdir -p ~/.config/systemd/user/
ln -s /path/to/ponkotsu_wcsc33/systemd/note-spsa-log.service ~/.config/systemd/user/
ln -s /path/to/ponkotsu_wcsc33/systemd/note-spsa-log.timer ~/.config/systemd/user/

# リロード・有効化・起動
systemctl --user daemon-reload
systemctl --user enable --now note-spsa-log.timer
```

## 確認

```bash
# timerの状態確認
systemctl --user status note-spsa-log.timer

# 手動実行テスト
systemctl --user start note-spsa-log.service

# ログ確認
journalctl --user -u note-spsa-log.service -f
```

## 停止・無効化

```bash
systemctl --user disable --now note-spsa-log.timer
```
