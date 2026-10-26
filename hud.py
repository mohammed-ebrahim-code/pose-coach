"""On-screen display (HUD) for Pose Coach.

Only drawing lives here. All the push-up logic (angles, rep counting,
form judging) stays in main.py.

Everything is sized from the frame height, so it looks the same on a
640x480 webcam and a 1280x720 one.
"""
import cv2 as cv

# Colors are BGR (blue, green, red), not RGB
WHITE = (240, 240, 240)
GREY = (170, 170, 170)
DARK = (22, 22, 22)
GREEN = (123, 198, 122)    # same green as the presentation (#7BC67A)
RED = (53, 107, 255)       # orange-red (#FF6B35)
AMBER = (65, 184, 245)     # warning yellow (#F5B841)
BLUE = (247, 195, 79)      # tracked arm (#4FC3F7)

FONT = cv.FONT_HERSHEY_DUPLEX
PANEL_ALPHA = 0.6          # how see-through the dark boxes are (0 = invisible, 1 = solid)


def scale_of(frame):
    """1.0 at 720p, smaller on smaller frames."""
    return frame.shape[0] / 720


def rounded_rect(img, x1, y1, x2, y2, color, radius):
    """Filled rectangle with rounded corners (OpenCV has no built-in one)."""
    x1, y1, x2, y2, r = int(x1), int(y1), int(x2), int(y2), int(radius)
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv.circle(img, (cx, cy), r, color, -1, cv.LINE_AA)


def text(img, words, x, y, size, color, weight=1, center=False):
    """putText with anti-aliasing. size and weight are scaled for you.
    x, y = bottom-left of the text, or bottom-centre when center=True."""
    s = scale_of(img)
    font_scale = size * s
    thickness = max(1, round(weight * s * 1.6))
    if center:
        (w, _), _ = cv.getTextSize(words, FONT, font_scale, thickness)
        x = x - w // 2
    cv.putText(img, words, (int(x), int(y)), FONT, font_scale, color, thickness, cv.LINE_AA)


def text_width(img, words, size, weight=1):
    s = scale_of(img)
    (w, _), _ = cv.getTextSize(words, FONT, size * s, max(1, round(weight * s * 1.6)))
    return w


def degree(img, words, x, y, size, color, weight=1, center=False):
    """Hershey fonts can't draw the degree sign, so draw a small circle after the number."""
    s = scale_of(img)
    w = text_width(img, words, size, weight)
    if center:
        x = x - w // 2
    text(img, words, x, y, size, color, weight)
    r = max(2, int(4 * s * size))
    cv.circle(img, (int(x + w + r + 2), int(y - size * 26 * s)), r, color, max(1, int(2 * s)), cv.LINE_AA)


def draw_skeleton(frame, points, connections, highlight=(), body_line=None, body_ok=True):
    """points: {landmark index: (x, y)} for the visible joints only.
    highlight: connections drawn thicker in blue (the arm being tracked).
    body_line: (shoulder, hip, ankle) pixel points, drawn green when straight, red when sagging."""
    s = scale_of(frame)
    for a, b in connections:
        if a in points and b in points:
            cv.line(frame, points[a], points[b], WHITE, max(2, int(3 * s)), cv.LINE_AA)
    for a, b in highlight:
        if a in points and b in points:
            cv.line(frame, points[a], points[b], BLUE, max(3, int(6 * s)), cv.LINE_AA)
    if body_line is not None:
        color = GREEN if body_ok else RED
        sh, hip, ank = body_line
        cv.line(frame, sh, hip, color, max(3, int(6 * s)), cv.LINE_AA)
        cv.line(frame, hip, ank, color, max(3, int(6 * s)), cv.LINE_AA)
    for p in points.values():
        cv.circle(frame, p, max(3, int(6 * s)), GREEN, -1, cv.LINE_AA)
        cv.circle(frame, p, max(3, int(6 * s)), DARK, 1, cv.LINE_AA)


def draw_hud(frame, info):
    """Draws every box and label. Call it AFTER the flip.

    info keys:
      good, bad          rep counts
      stage              "up" or "down"
      fps                number
      angle              elbow angle being counted, or None
      body_angle         current body angle, or None
      sag_angle, down, up    thresholds (for colours and the gauge)
      feedback           "good", "bad" or None (shown for 1 second by main.py)
      warning            text to show when the body isn't visible, or None
      elbow_pos          (x, y) of the tracked elbow on the shown (flipped) frame, or None
    """
    s = scale_of(frame)
    h, w = frame.shape[:2]
    m = int(24 * s)  # margin

    # ---- 1. dark boxes, all blended in one go ----
    overlay = frame.copy()
    panel = (m, m, m + int(330 * s), m + int(196 * s))
    rounded_rect(overlay, *panel, DARK, 16 * s)

    pill_w, pill_h = int(170 * s), int(52 * s)
    pill = (w - m - pill_w, m, w - m, m + pill_h)
    rounded_rect(overlay, *pill, DARK, pill_h // 2)

    gx1, gx2 = w - m - int(70 * s), w - m - int(46 * s)
    gy1, gy2 = m + int(150 * s), h - int(150 * s)
    rounded_rect(overlay, gx1 - int(40 * s), gy1 - int(92 * s), gx2 + int(30 * s), gy2 + int(36 * s), DARK, 14 * s)

    banner = None
    if info["feedback"] is not None or info["warning"] is not None:
        words = {"good": "GOOD REP", "bad": "HIPS SAGGING"}.get(info["feedback"], info["warning"])
        size = 1.2 if info["feedback"] else 0.85
        bw = text_width(frame, words, size, 2) + int(64 * s)
        bh = int(70 * s)
        bx = w // 2 - bw // 2
        by = m + int(214 * s)  # upper middle: in a side-view push-up the body fills the lower half
        color = {"good": GREEN, "bad": RED}.get(info["feedback"], AMBER)
        banner = (bx, by, bx + bw, by + bh, words, size, color)
        rounded_rect(overlay, bx, by, bx + bw, by + bh, color, bh // 2)

    if info["elbow_pos"] is not None and info["angle"] is not None:
        ex, ey = info["elbow_pos"]
        lx, ly = ex + int(18 * s), ey - int(18 * s)
        rounded_rect(overlay, lx, ly - int(36 * s), lx + int(96 * s), ly + int(8 * s), DARK, 10 * s)

    cv.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0, frame)

    # ---- 2. rep panel (top-left) ----
    px, py = panel[0] + int(22 * s), panel[1]
    text(frame, "PUSH-UPS", px, py + int(36 * s), 0.6, GREY)
    total = info["good"] + info["bad"]
    text(frame, str(total), px, py + int(128 * s), 3.0, WHITE, 3)
    cx = px + int(150 * s)
    text(frame, f"GOOD {info['good']}", cx, py + int(78 * s), 0.8, GREEN, 2)
    text(frame, f"BAD  {info['bad']}", cx, py + int(122 * s), 0.8, RED, 2)
    if info["body_angle"] is not None:
        ok = info["body_angle"] >= info["sag_angle"]
        degree(frame, f"Body line {info['body_angle']}", px, py + int(176 * s), 0.6, GREEN if ok else RED)
    else:
        text(frame, "Body line --", px, py + int(176 * s), 0.6, GREY)

    # ---- 3. stage pill (top-right) ----
    stage = info["stage"].upper()
    dot = GREEN if stage == "UP" else BLUE
    cv.circle(frame, (pill[0] + int(30 * s), m + pill_h // 2), int(8 * s), dot, -1, cv.LINE_AA)
    text(frame, stage, pill[0] + int(50 * s), m + pill_h // 2 + int(11 * s), 0.9, WHITE, 2)

    # ---- 4. elbow angle gauge (right side) ----
    lo, hi = 40, 180                       # gauge range in degrees
    def gy(a):                             # angle -> y on the gauge
        a = max(lo, min(hi, a))
        return int(gy2 - (a - lo) / (hi - lo) * (gy2 - gy1))
    text(frame, "ELBOW", (gx1 + gx2) // 2 - int(5 * s), gy1 - int(62 * s), 0.55, GREY, center=True)
    if info["angle"] is not None:
        degree(frame, str(info["angle"]), (gx1 + gx2) // 2 - int(8 * s), gy1 - int(20 * s), 0.8, WHITE, 2, center=True)
    else:
        text(frame, "--", (gx1 + gx2) // 2 - int(5 * s), gy1 - int(20 * s), 0.8, GREY, 2, center=True)
    cv.rectangle(frame, (gx1, gy1), (gx2, gy2), (70, 70, 70), -1)
    if info["angle"] is not None:
        a = info["angle"]
        fill = GREEN if a > info["up"] else (BLUE if a >= info["down"] else RED)
        cv.rectangle(frame, (gx1, gy(a)), (gx2, gy2), fill, -1)
    for mark, label in ((info["up"], "UP"), (info["down"], "DOWN")):
        y = gy(mark)
        cv.line(frame, (gx1 - int(8 * s), y), (gx2 + int(8 * s), y), WHITE, max(1, int(2 * s)), cv.LINE_AA)
        text(frame, str(mark), gx1 - int(10 * s) - text_width(frame, str(mark), 0.5), y + int(6 * s), 0.5, WHITE)

    # ---- 5. feedback / warning banner (upper middle) ----
    if banner is not None:
        bx1, by1, bx2, by2, words, size, color = banner
        ink = DARK if color in (GREEN, AMBER) else WHITE
        text(frame, words, (bx1 + bx2) // 2, (by1 + by2) // 2 + int(14 * s * size), size, ink, 2, center=True)

    # ---- 6. elbow label next to the joint ----
    if info["elbow_pos"] is not None and info["angle"] is not None:
        ex, ey = info["elbow_pos"]
        degree(frame, str(info["angle"]), ex + int(30 * s), ey - int(20 * s), 0.75, WHITE, 2)

    # ---- 7. key hints (bottom-left) + FPS (bottom-right) ----
    text(frame, "Q quit   R reset", m, h - m, 0.5, GREY)
    fps_words = f"FPS {info['fps']}"
    text(frame, fps_words, w - m - text_width(frame, fps_words, 0.5), h - m, 0.5, GREY)