#!/usr/bin/env python3
import ast
import os
import re
import sys


def extract_section(block_text, start_token, end_tokens):
    start = block_text.find(start_token)
    if start == -1:
        return None

    end_positions = []
    for token in end_tokens:
        pos = block_text.find(token, start + len(start_token))
        if pos != -1:
            end_positions.append(pos)
    end = min(end_positions) if end_positions else len(block_text)
    return block_text[start:end]


def extract_section_pct(section_text):
    if not section_text:
        return None, False, None, None

    # Pick only overall match summary immediately after progress line.
    games = re.findall(
        r"(\d+) of (\d+) games finished\.\n(?!Black vs White:)(?!.* playing (?:Black|White):).+ vs .+: \d+-\d+-\d+ \(([\d.]+)%\)",
        section_text,
        re.MULTILINE,
    )
    if not games:
        return None, False, None, None

    finished, total, pct = games[-1]
    finished_i = int(finished)
    total_i = int(total)
    return float(pct), finished_i >= total_i, finished_i, total_i


def extract_last_completed_wr(log_text):
    wr_sep = re.findall(r"win_rate\+=([\d.]+), win_rate-=([\d.]+)", log_text)
    wr_pm = re.findall(r"win_rate\(\+vs-\)=([\d.]+)", log_text)

    plus = None
    minus = None
    pm = None
    if wr_sep:
        p, m = wr_sep[-1]
        plus = float(p) * 100.0
        minus = float(m) * 100.0
    if wr_pm:
        pm = float(wr_pm[-1]) * 100.0
    return plus, minus, pm


def build_note_text(log_text):
    blocks = re.split(r"={50,}", log_text)
    completed = sum(1 for b in blocks if re.search(r"Updated theta", b))

    iteration_blocks = [b for b in blocks if re.search(r"Iteration \d+/\d+", b)]
    current_block = iteration_blocks[-1] if iteration_blocks else log_text

    inits = re.findall(r"Initial params: ({.*?})", log_text)
    init_p = ast.literal_eval(inits[-1]) if inits else {}
    thetas = re.findall(r"Updated theta = ({.*?})", log_text)
    current = ast.literal_eval(thetas[-1]) if thetas else init_p

    plus_section = extract_section(
        current_block,
        "Playing theta+ match",
        ["Playing theta- match", "Playing theta+ vs theta- match"],
    )
    minus_section = extract_section(
        current_block,
        "Playing theta- match",
        ["Playing theta+ vs theta- match"],
    )
    pm_section = extract_section(current_block, "Playing theta+ vs theta- match", [])

    plus_pct, plus_done, plus_finished, plus_total = extract_section_pct(plus_section)
    minus_pct, minus_done, minus_finished, minus_total = extract_section_pct(minus_section)
    pm_pct, pm_done, pm_finished, pm_total = extract_section_pct(pm_section)

    base_plus, base_minus, base_pm = extract_last_completed_wr(log_text)

    phase_positions = []
    for phase_name, token in [
        ("plus", "Playing theta+ match"),
        ("minus", "Playing theta- match"),
        ("pm", "Playing theta+ vs theta- match"),
    ]:
        pos = current_block.rfind(token)
        if pos != -1:
            phase_positions.append((pos, phase_name))

    current_phase = max(phase_positions)[1] if phase_positions else None

    disp_plus = plus_pct if plus_pct is not None else base_plus
    disp_minus = minus_pct if minus_pct is not None else base_minus
    disp_pm = pm_pct if pm_pct is not None else base_pm
    plus_live = False
    minus_live = False
    pm_live = False

    if current_phase == "plus" and plus_pct is not None and not plus_done:
        disp_plus = plus_pct
        plus_live = True
    elif current_phase == "minus":
        if plus_pct is not None:
            disp_plus = plus_pct
        if minus_pct is not None:
            disp_minus = minus_pct
            minus_live = not minus_done
    elif current_phase == "pm" and pm_pct is not None:
        disp_pm = pm_pct
        pm_live = not pm_done

    lines = ["SPSA最適化ログ", f"完了: {completed} iterations"]
    if disp_plus is not None:
        if plus_live and plus_finished is not None and plus_total is not None:
            suffix = f" (進行中: {plus_finished}/{plus_total})"
        else:
            suffix = " (進行中)" if plus_live else ""
        lines.append(f"最新 WR+: {disp_plus:.1f}%{suffix}")
    if disp_minus is not None:
        if minus_live and minus_finished is not None and minus_total is not None:
            suffix = f" (進行中: {minus_finished}/{minus_total})"
        else:
            suffix = " (進行中)" if minus_live else ""
        lines.append(f"最新 WR-: {disp_minus:.1f}%{suffix}")
    if disp_pm is not None:
        if pm_live and pm_finished is not None and pm_total is not None:
            suffix = f" (進行中: {pm_finished}/{pm_total})"
        else:
            suffix = " (進行中)" if pm_live else ""
        lines.append(f"最新 WR+vs-: {disp_pm:.1f}%{suffix}")
    lines.append(f"現在θ: {current}")

    return "\n".join(lines)


def main():
    log_file = os.environ.get("LOG_FILE")
    if not log_file:
        print("LOG_FILE is required", file=sys.stderr)
        return 1

    with open(log_file) as f:
        text = f.read()

    print(build_note_text(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
