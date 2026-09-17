# -*- coding: utf-8 -*-
"""生成 3 分钟 Demo 视频（1920x1080 / 30fps / H.264 + AAC，带中文旁白）。

用法：
    python make_video.py --dry-run        # 只打印时间轴，不出图
    python make_video.py                  # 渲染并编码
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AUDIO = os.path.join(HERE, "audio")
FIGDIR = os.path.join(ROOT, "results", "figures")
OUT_SILENT = os.path.join(HERE, "search_r1_demo_silent.mp4")
OUT_FINAL = os.path.join(HERE, "search_r1_demo.mp4")
AUDIO_MIX = os.path.join(HERE, "_narration_mix.wav")

W, H, FPS = 1920, 1080, 30

BG = (247, 248, 250)
WHITE = (255, 255, 255)
INK = (26, 28, 33)
GREY = (120, 126, 138)
LIGHT = (222, 226, 233)
BLUE = (47, 111, 181)
RED = (192, 57, 43)
GREEN = (46, 139, 87)
HILITE = (255, 245, 235)

F_REG = r"C:\Windows\Fonts\msyh.ttc"
F_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size, bold=False):
    try:
        return ImageFont.truetype(F_BOLD if bold else F_REG, size)
    except OSError:
        return ImageFont.load_default()


def text_w(draw, s, f):
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0]


def ease(t):
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def clamp01(t):
    return max(0.0, min(1.0, t))


# ---------------------------------------------------------------------------
# 帧渲染
# ---------------------------------------------------------------------------
def new_frame():
    return Image.new("RGB", (W, H), BG)


def header(d, sec, idx, total):
    d.rectangle([0, 0, W, 8], fill=BLUE)
    d.text((90, 58), sec["title"], font=font(50, True), fill=INK)
    d.text((92, 128), sec.get("subtitle", ""), font=font(26), fill=GREY)
    d.line([(90, 178), (W - 90, 178)], fill=LIGHT, width=2)
    # 右上角序号
    d.text((W - 90, 62), f"{idx:02d} / {total}", font=font(24, True), fill=LIGHT, anchor="ra")


def footer(d, p_overall):
    d.rectangle([90, H - 46, W - 90, H - 40], fill=LIGHT)
    d.rectangle([90, H - 46, int(90 + (W - 180) * clamp01(p_overall)), H - 40], fill=BLUE)


def draw_bullets(d, sec, p):
    y = 250
    items = sec["bullets"]
    for i, it in enumerate(items):
        start = i * 0.16
        a = ease((p - start) / 0.22)
        if a <= 0:
            continue
        dy = int((1 - a) * 22)
        col = BLUE if i == 0 else INK
        f = font(34, i == 0)
        d.ellipse([92 + 0, y + 12 + dy, 92 + 12, y + 24 + dy], fill=BLUE)
        d.text((128, y + dy), it, font=f, fill=col)
        y += 78
    if p > 0.72:
        a = ease((p - 0.72) / 0.2)
        d.rectangle([90, y + 20, W - 90, y + 118], fill=HILITE, outline=(240, 210, 180), width=2)
        d.text((124, y + 46), "只用 outcome reward：不加格式奖励、不训 reward model",
               font=font(30, True), fill=(150, 80, 30))


def draw_tree(d, sec, p):
    lines = sec["tree"]
    y = 232
    for i, ln in enumerate(lines):
        a = ease((p - i * 0.09) / 0.16)
        if a <= 0:
            continue
        dy = int((1 - a) * 16)
        name, _, desc = ln.partition("  ")
        d.text((110, y + dy), name, font=font(30, True), fill=BLUE)
        if desc:
            d.text((560, y + dy), desc.strip(), font=font(28), fill=INK)
        y += 66
    if p > 0.68:
        a = ease((p - 0.68) / 0.2)
        y2 = y + 26
        d.rectangle([90, y2, W - 90, y2 + 92], fill=(238, 244, 252), outline=(180, 205, 235), width=2)
        d.text((124, y2 + 20), "$ python run_all.py", font=font(32, True), fill=BLUE)
        d.text((124, y2 + 60), "8 步全部通过 · 6.9 秒 · 不需要 GPU → figures / dryrun / failure_attribution",
               font=font(26), fill=GREY)


def draw_trace(d, sec, p):
    trace = sec["trace"]
    y = 240
    colors = {"<think>": BLUE, "<search>": RED, "<information>": (120, 126, 138),
              "<answer>": GREEN}
    for i, (tag, desc) in enumerate(trace):
        a = ease((p - i * 0.13) / 0.18)
        if a <= 0:
            continue
        dy = int((1 - a) * 18)
        c = colors.get(tag, INK)
        tw = text_w(d, tag, font(34, True))
        d.rounded_rectangle([110, y + dy, 110 + tw + 40, y + 58 + dy], radius=10, fill=c)
        d.text((130, y + 10 + dy), tag, font=font(34, True), fill=WHITE)
        d.text((110 + tw + 72, y + 12 + dy), desc, font=font(30), fill=INK)
        y += 92
    if p > 0.85:
        a = ease((p - 0.85) / 0.15)
        d.rectangle([90, y + 4, W - 90, y + 78], fill=(238, 244, 252), outline=(180, 205, 235), width=2)
        d.text((124, y + 24), "灰色块 = 环境注入，不计入 loss（retrieved token masking）",
               font=font(28, True), fill=BLUE)


def draw_pitfall(d, sec, p):
    pf = sec["pitfall"]
    d.text((110, 216), pf["note"], font=font(30, True), fill=RED)
    # wrong
    d.rectangle([90, 288, W - 90, 400], fill=(253, 240, 240), outline=(230, 190, 190), width=2)
    d.text((122, 306), "✗ 错误", font=font(28, True), fill=RED)
    d.text((122, 348), pf["wrong"], font=font(28), fill=(140, 40, 40))
    # right
    d.rectangle([90, 424, W - 90, 536], fill=(238, 249, 242), outline=(180, 220, 195), width=2)
    d.text((122, 442), "✓ 正确", font=font(28, True), fill=GREEN)
    d.text((122, 484), pf["right"], font=font(28), fill=(30, 100, 60))
    # 结果
    if p > 0.55:
        a = ease((p - 0.55) / 0.2)
        d.rectangle([90, 570, W - 90, 700], fill=HILITE, outline=(240, 210, 180), width=2)
        d.text((122, 592), "后果：RAG 基线  avg_search = 0，EM = 0.00 —— 代码不报错，结果全错",
               font=font(30, True), fill=(150, 80, 30))
        d.text((122, 640), "另一个坑：多轮检索召回变多后聚合题反而变差（0.613 → 0.543），加约束过滤才回到 0.788",
               font=font(26), fill=(150, 100, 50))
    if p > 0.8:
        d.rectangle([90, 726, W - 90, 830], fill=(238, 244, 252), outline=(180, 205, 235), width=2)
        d.text((122, 750), "对应论文 Appendix G：top-k 3 → 5，EM 从 0.431 掉到 0.400 —— 召回更多 ≠ 更好",
               font=font(28, True), fill=BLUE)


def draw_table(d, sec, p):
    t = sec["table"]
    cols = [110, 520, 980]
    d.rectangle([90, 220, W - 90, 292], fill=BLUE)
    for c, htxt in zip(cols, t["headers"]):
        d.text((c + 12, 240), htxt, font=font(30, True), fill=WHITE)
    y = 292
    for i, row in enumerate(t["rows"]):
        a = ease((p - 0.1 - i * 0.13) / 0.18)
        if a <= 0:
            continue
        h2 = 84
        d.rectangle([90, y, W - 90, y + h2], fill=WHITE if i % 2 == 0 else (252, 253, 254),
                    outline=LIGHT, width=1)
        for c, cell in zip(cols, row):
            f = font(28, c == cols[0])
            d.text((c + 12, y + 24), cell, font=f, fill=INK if c != cols[0] else BLUE)
        y += h2
    if p > 0.78:
        d.rectangle([90, y + 26, W - 90, y + 128], fill=HILITE, outline=(240, 210, 180), width=2)
        d.text((124, y + 50), "自建评测集 ContentSearch-Bench：212 题，题目由语料构造、答案由代码算出",
               font=font(30, True), fill=(150, 80, 30))
        d.text((124, y + 90), "含 16 道不可答题 —— 三个系统在上面全部 0 分", font=font(26), fill=(150, 100, 50))


def draw_figure(d, sec, p):
    fig_path = os.path.join(FIGDIR, sec["figure"])
    panel_x, panel_w = 90, 1180
    card = [panel_x, 220, panel_x + panel_w, 950]
    d.rounded_rectangle(card, radius=14, fill=WHITE, outline=LIGHT, width=2)
    if os.path.exists(fig_path):
        im = Image.open(fig_path).convert("RGB")
        # Ken Burns：轻微放大
        z = 1.0 + 0.045 * p
        tw, th = int((card[2] - card[0]) * z), int((card[3] - card[1]) * z)
        im = im.resize((tw, th), Image.LANCZOS)
        cx, cy = (tw - (card[2] - card[0])) // 2, (th - (card[3] - card[1])) // 2
        im = im.crop((cx, cy, cx + (card[2] - card[0]), cy + (card[3] - card[1])))
        d._im = im  # 让外层合成
        return (card, im)
    return (card, None)


def draw_highlight(d, sec, p):
    x = 1300
    y = 240
    d.rounded_rectangle([x, y, W - 90, 950], radius=14, fill=WHITE, outline=LIGHT, width=2)
    d.text((x + 34, y + 34), "关键数字", font=font(30, True), fill=BLUE)
    yy = y + 96
    for i, row in enumerate(sec.get("highlight", [])):
        a = ease((p - 0.12 - i * 0.11) / 0.18)
        if a <= 0:
            continue
        dy = int((1 - a) * 14)
        name, val = row[0], row[1]
        extra = row[2] if len(row) > 2 else ""
        d.text((x + 34, yy + dy), name, font=font(27), fill=GREY)
        d.text((x + 34, yy + 36 + dy), val, font=font(46, True),
               fill=RED if i == len(sec["highlight"]) - 1 else INK)
        if extra and extra != "—":
            d.text((x + 34 + text_w(d, val, font(46, True)) + 18, yy + 52 + dy), extra,
                   font=font(26), fill=GREY)
        yy += 118


def draw_title_card(d, sec, p):
    a = ease(p / 0.35)
    d.rectangle([0, 0, W, H], fill=BG)
    d.rectangle([0, 0, W, 10], fill=BLUE)
    d.text((W // 2, 380 - int((1 - a) * 30)), sec["title"], font=font(78, True), fill=INK,
           anchor="ma")
    d.text((W // 2, 500), sec["subtitle"], font=font(36), fill=GREY, anchor="ma")
    if p > 0.35:
        a2 = ease((p - 0.35) / 0.25)
        d.line([(W // 2 - 260, 580), (W // 2 + 260, 580)], fill=LIGHT, width=3)
        d.text((W // 2, 640), "github.com/Belovedarling/search-r1-repro", font=font(34, True),
               fill=BLUE, anchor="ma")
        d.text((W // 2, 700), "Search-R1: Training LLMs to Reason and Leverage Search Engines with RL",
               font=font(26), fill=GREY, anchor="ma")
        d.text((W // 2, 742), "Bowen Jin et al., COLM 2025 · arXiv:2503.09516",
               font=font(26), fill=GREY, anchor="ma")


def draw_end(d, sec, p):
    d.rectangle([0, 0, W, H], fill=BG)
    d.text((W // 2, 420), sec["title"], font=font(96, True), fill=BLUE, anchor="ma")
    d.text((W // 2, 540), sec["subtitle"], font=font(40, True), fill=INK, anchor="ma")
    if p > 0.4:
        d.text((W // 2, 640), "复现报告 · 偏差分析 · 失败案例库 · 自建评测集",
               font=font(30), fill=GREY, anchor="ma")


def render_section(sec, idx, total, p, p_overall):
    img = new_frame()
    d = ImageDraw.Draw(img)
    kind = sec["kind"]

    if kind == "title":
        draw_title_card(d, sec, p)
    elif kind == "end":
        draw_end(d, sec, p)
    elif kind == "figure":
        header(d, sec, idx, total)
        card, im = draw_figure(d, sec, p)
        draw_highlight(d, sec, p)
        if im is not None:
            img.paste(im, (card[0], card[1]))
    else:
        header(d, sec, idx, total)
        if kind == "bullets":
            draw_bullets(d, sec, p)
        elif kind == "tree":
            draw_tree(d, sec, p)
        elif kind == "trace":
            draw_trace(d, sec, p)
        elif kind == "pitfall":
            draw_pitfall(d, sec, p)
        elif kind == "table":
            draw_table(d, sec, p)
    footer(d, p_overall)

    # 段落淡入淡出（白场转场）
    fade = 0.35
    if p < fade:
        w_ = Image.new("RGB", (W, H), BG)
        img = Image.blend(w_, img, ease(p / fade))
    elif p > 1 - fade:
        w_ = Image.new("RGB", (W, H), BG)
        img = Image.blend(img, w_, ease((p - (1 - fade)) / fade))
    return img


# ---------------------------------------------------------------------------
# 音频
# ---------------------------------------------------------------------------
def wav_info(path):
    with wave.open(path, "rb") as f:
        return f.getnframes() / f.getframerate(), f.getframerate(), f.getnchannels(), f.getsampwidth()


def build_audio(sections):
    """拼接各段旁白（段间留静音），返回 (总时长, 每段时长, 采样率, 声道, 位宽)"""
    sr = ch = sw = None
    datas, durs = [], []
    for s in sections:
        p = os.path.join(AUDIO, s["id"] + ".wav")
        dur, sr, ch, sw = wav_info(p)
        with wave.open(p, "rb") as f:
            datas.append(f.readframes(f.getnframes()))
        durs.append(dur)
    pad = int(sr * 0.55)
    silence = b"\x00" * (pad * ch * sw)
    total = 0.0
    sec_dur = []
    for i, (d, data) in enumerate(zip(durs, datas)):
        # 段落时长 = 音频 + 尾部留白（至少 6.5 秒，保证可读）
        sd = max(d + 0.55, 6.5)
        sec_dur.append(sd)
        total += sd
    with wave.open(AUDIO_MIX, "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(sw)
        out.setframerate(sr)
        for i, data in enumerate(datas):
            out.writeframes(data)
            tail = int((sec_dur[i] - durs[i]) * sr - pad // (ch * sw))
            if tail > 0:
                out.writeframes(b"\x00" * (tail * ch * sw))
            out.writeframes(silence)
    return total, sec_dur


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "narration.json"), encoding="utf-8"))
    sections = cfg["sections"]

    total, sec_dur = build_audio(sections)
    print(f"总时长 {total:.1f}s  ({int(total // 60)}分{total % 60:.0f}秒)")
    for s, d in zip(sections, sec_dur):
        print(f"  {s['id']} {s['kind']:8s} {d:6.2f}s  {s['title']}")
    if args.dry_run:
        return

    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    nframes = int(total * FPS)
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
           "-crf", "20", "-pix_fmt", "yuv420p", OUT_SILENT]
    print("\nffmpeg:", " ".join(cmd[:6]), "...")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)

    fi = 0
    cache = {}
    for si, (sec, dur) in enumerate(zip(sections, sec_dur)):
        nf = max(1, int(round(dur * FPS)))
        for k in range(nf):
            p_loc = (k + 0.5) / nf
            p_glob = (fi + k) / max(nframes, 1)
            # 静态段缓存：揭示动画结束后复用上一帧
            key = None
            if p_loc > 0.92 and si in cache:
                img = cache[si]
            else:
                img = render_section(sec, si + 1, len(sections), p_loc, p_glob)
                if p_loc > 0.92:
                    cache[si] = img
            p.stdin.write(img.tobytes())
            if (fi + k) % 300 == 0:
                print(f"  frame {fi+k}/{nframes}", flush=True)
        fi += nf

    p.stdin.close()
    err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0:
        print("ffmpeg 编码失败:\n", err[-1500:])
        return 1

    if args.no_audio:
        print("silent:", OUT_SILENT)
        return 0

    cmd2 = [ff, "-y", "-i", OUT_SILENT, "-i", AUDIO_MIX,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", OUT_FINAL]
    r = subprocess.run(cmd2, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("混音失败:\n", (r.stderr or "")[-1500:])
        return 1
    print("\n完成:", OUT_FINAL)
    print("大小: %.1f MB" % (os.path.getsize(OUT_FINAL) / 1024 / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
