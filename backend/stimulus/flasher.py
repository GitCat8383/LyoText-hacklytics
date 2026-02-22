"""Pygame-based P300 stimulus flashcard window (patient-facing).

Each word is highlighted sequentially for FLASHCARD_DURATION_MS (4s default),
cycling through FLASHCARD_LOOPS (2 default). A progress bar drains on the
active tile. Words update instantly when the grid changes.
"""

from __future__ import annotations

import time
from multiprocessing import Process, Queue

import config

# ── Colors ────────────────────────────────────────────────────
BG_COLOR = (18, 18, 24)
TILE_COLOR = (35, 39, 52)
TILE_ACTIVE = (255, 220, 80)
TILE_SELECTED = (80, 200, 120)
TILE_CALIBRATION = (255, 100, 100)
TEXT_COLOR = (220, 220, 230)
TEXT_ACTIVE = (18, 18, 24)
HEADER_COLOR = (160, 160, 180)
STATUS_COLOR = (100, 100, 120)
PROGRESS_BG = (60, 60, 80)
PROGRESS_FG = (255, 255, 255, 160)
LOOP_COLOR = (120, 180, 255)
SENTENCE_BG = (30, 35, 50)
SENTENCE_WORD_BG = (60, 70, 100)
SENTENCE_WORD_TEXT = (220, 225, 240)
SENTENCE_LABEL_COLOR = (100, 110, 140)

WINDOW_W, WINDOW_H = 1280, 900
TILE_MARGIN = 20
COLS, ROWS = 3, 2
HEADER_H = 90
SENTENCE_H = 80
FOOTER_H = 70


def _run_flasher(cmd_queue: Queue, event_queue: Queue) -> None:
    import pygame
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Hacklytic - P300 Stimulus")
    clock = pygame.time.Clock()

    font_word = pygame.font.SysFont("Helvetica", 36, bold=True)
    font_header = pygame.font.SysFont("Helvetica", 24, bold=True)
    font_status = pygame.font.SysFont("Helvetica", 18)
    font_loop = pygame.font.SysFont("Helvetica", 16, bold=True)
    font_index = pygame.font.SysFont("Helvetica", 14)

    font_sentence_label = pygame.font.SysFont("Helvetica", 14, bold=True)
    font_sentence_word = pygame.font.SysFont("Helvetica", 22, bold=True)

    words: list[str] = ["Waiting..."] * config.NUM_PHRASES
    sentence_words: list[str] = []
    highlight_index = -1
    highlight_color = TILE_SELECTED
    calibration_target = -1
    status_text = "Waiting for connection..."

    # Flashcard animation state
    flashing = False
    flash_index = -1
    flash_step = 0
    flash_loop = 0
    flash_start_time = 0.0
    total_loops = config.FLASHCARD_LOOPS
    duration_ms = config.FLASHCARD_DURATION_MS

    def _tile_rects() -> list:
        usable_w = WINDOW_W - TILE_MARGIN * (COLS + 1)
        top = HEADER_H + SENTENCE_H + TILE_MARGIN
        usable_h = WINDOW_H - top - FOOTER_H - TILE_MARGIN * (ROWS + 1)
        tw = usable_w // COLS
        th = usable_h // ROWS
        rects = []
        for row in range(ROWS):
            for col in range(COLS):
                x = TILE_MARGIN + col * (tw + TILE_MARGIN)
                y = top + TILE_MARGIN + row * (th + TILE_MARGIN)
                rects.append(pygame.Rect(x, y, tw, th))
        return rects

    tile_rects = _tile_rects()
    PROGRESS_BAR_H = 6

    def _draw():
        screen.fill(BG_COLOR)

        # Header
        header_text = "Focus on the highlighted word"
        header = font_header.render(header_text, True, HEADER_COLOR)
        screen.blit(header, (WINDOW_W // 2 - header.get_width() // 2, 20))

        # Sentence display
        sent_y = HEADER_H
        sent_rect = pygame.Rect(TILE_MARGIN, sent_y, WINDOW_W - 2 * TILE_MARGIN, SENTENCE_H - 5)
        pygame.draw.rect(screen, SENTENCE_BG, sent_rect, border_radius=14)

        label_surf = font_sentence_label.render("SENTENCE:", True, SENTENCE_LABEL_COLOR)
        screen.blit(label_surf, (sent_rect.x + 14, sent_rect.y + 8))

        if sentence_words:
            word_x = sent_rect.x + 14
            word_y = sent_rect.y + 30
            for sw in sentence_words:
                sw_surf = font_sentence_word.render(sw, True, SENTENCE_WORD_TEXT)
                sw_w = sw_surf.get_width() + 16
                if word_x + sw_w > sent_rect.right - 10:
                    break
                pill_rect = pygame.Rect(word_x, word_y, sw_w, 32)
                pygame.draw.rect(screen, SENTENCE_WORD_BG, pill_rect, border_radius=8)
                screen.blit(sw_surf, (word_x + 8, word_y + 4))
                word_x += sw_w + 6
        else:
            hint = font_sentence_label.render("Focus on words to build your sentence...", True, (80, 80, 100))
            screen.blit(hint, (sent_rect.x + 14, sent_rect.y + 36))

        # Loop / progress indicator
        if flashing:
            total_steps = len(words) * total_loops
            current_step = flash_loop * len(words) + (flash_step + 1)
            loop_text = f"Loop {flash_loop + 1}/{total_loops}   •   Word {flash_step + 1}/{len(words)}   •   Step {current_step}/{total_steps}"
            loop_surf = font_loop.render(loop_text, True, LOOP_COLOR)
            screen.blit(loop_surf, (WINDOW_W // 2 - loop_surf.get_width() // 2, 52))

        # Tiles
        for i, rect in enumerate(tile_rects):
            if i >= len(words):
                break

            is_active = (i == flash_index and flashing)
            is_selected = (i == highlight_index)
            is_cal = (i == calibration_target)

            if is_active:
                color = TILE_ACTIVE
                text_c = TEXT_ACTIVE
            elif is_selected:
                color = highlight_color
                text_c = TEXT_ACTIVE
            elif is_cal:
                color = TILE_CALIBRATION
                text_c = TEXT_ACTIVE
            else:
                color = TILE_COLOR
                text_c = TEXT_COLOR

            # Scale up active tile slightly
            draw_rect = rect
            if is_active:
                inflate = 8
                draw_rect = rect.inflate(inflate, inflate)

            pygame.draw.rect(screen, color, draw_rect, border_radius=16)

            # Word text (bigger for active)
            if is_active:
                active_font = pygame.font.SysFont("Helvetica", 48, bold=True)
                text_surf = active_font.render(words[i], True, text_c)
            else:
                text_surf = font_word.render(words[i], True, text_c)
            tx = draw_rect.centerx - text_surf.get_width() // 2
            ty = draw_rect.centery - text_surf.get_height() // 2
            screen.blit(text_surf, (tx, ty))

            # Index number (top-left corner)
            idx_surf = font_index.render(str(i + 1), True,
                                         (80, 80, 80) if is_active else (70, 70, 90))
            screen.blit(idx_surf, (draw_rect.x + 10, draw_rect.y + 8))

            # Progress bar on active tile (drains left to right)
            if is_active and flashing:
                elapsed = (time.time() - flash_start_time) * 1000
                progress = max(0.0, 1.0 - elapsed / duration_ms)
                bar_y = draw_rect.bottom - PROGRESS_BAR_H - 4
                bar_w = draw_rect.width - 20
                bar_x = draw_rect.x + 10

                pygame.draw.rect(screen, PROGRESS_BG,
                                 (bar_x, bar_y, bar_w, PROGRESS_BAR_H),
                                 border_radius=3)
                if progress > 0:
                    pygame.draw.rect(screen, (255, 255, 255),
                                     (bar_x, bar_y, int(bar_w * progress), PROGRESS_BAR_H),
                                     border_radius=3)

        # Footer status
        status_surf = font_status.render(status_text, True, STATUS_COLOR)
        screen.blit(status_surf, (TILE_MARGIN, WINDOW_H - FOOTER_H + 20))

        pygame.display.flip()

    def _start_flash_sequence():
        nonlocal flashing, flash_index, flash_step, flash_loop, flash_start_time
        flashing = True
        flash_step = 0
        flash_loop = 0
        flash_index = 0
        flash_start_time = time.time()

    def _advance_flash():
        nonlocal flashing, flash_index, flash_step, flash_loop, flash_start_time

        flash_step += 1
        if flash_step >= len(words):
            flash_step = 0
            flash_loop += 1
            if flash_loop >= total_loops:
                flashing = False
                flash_index = -1
                event_queue.put({
                    "event": "flash_cycle_complete",
                    "timestamp": time.time(),
                })
                return

        flash_index = flash_step
        flash_start_time = time.time()

        event_queue.put({
            "event": "stimulus_onset",
            "phrase_index": flash_index,
            "timestamp": flash_start_time,
        })

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # Process commands from backend
        while not cmd_queue.empty():
            try:
                cmd = cmd_queue.get_nowait()
            except Exception:
                break

            action = cmd.get("cmd")
            if action == "set_phrases":
                words = cmd["phrases"]
                status_text = f"Words: {', '.join(words)}"
                tile_rects = _tile_rects()
                if flashing:
                    _start_flash_sequence()
            elif action == "set_sentence":
                sentence_words = cmd.get("words", [])
            elif action == "start_flash":
                highlight_index = -1
                calibration_target = -1
                _start_flash_sequence()
                status_text = "Flashing..."
            elif action == "stop_flash":
                flashing = False
                flash_index = -1
                status_text = "Stopped"
            elif action == "highlight":
                highlight_index = cmd.get("index", -1)
                color_name = cmd.get("color", "green")
                highlight_color = TILE_SELECTED if color_name == "green" else TILE_CALIBRATION
                flash_index = -1
                flashing = False
                if 0 <= highlight_index < len(words):
                    status_text = f"Selected: {words[highlight_index]}"
            elif action == "reset_highlight":
                highlight_index = -1
            elif action == "show_calibration":
                calibration_target = cmd.get("target_index", -1)
                if 0 <= calibration_target < len(words):
                    status_text = f"FOCUS on: {words[calibration_target]}"
            elif action == "quit":
                running = False

        # Advance flashcard animation
        if flashing and len(words) > 0:
            elapsed = (time.time() - flash_start_time) * 1000
            if elapsed >= duration_ms:
                _advance_flash()

        _draw()
        clock.tick(60)

    pygame.quit()


class StimulusFlasher:

    def __init__(self) -> None:
        self._process: Process | None = None
        self.cmd_queue: Queue = Queue()
        self.event_queue: Queue = Queue()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._process = Process(
            target=_run_flasher,
            args=(self.cmd_queue, self.event_queue),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        if self.is_running:
            self.cmd_queue.put({"cmd": "quit"})
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()

    def set_phrases(self, phrases: list[str]) -> None:
        self.cmd_queue.put({"cmd": "set_phrases", "phrases": phrases})

    def set_sentence(self, words: list[str]) -> None:
        self.cmd_queue.put({"cmd": "set_sentence", "words": words})

    def start_flash(self) -> None:
        self.cmd_queue.put({"cmd": "start_flash"})

    def stop_flash(self) -> None:
        self.cmd_queue.put({"cmd": "stop_flash"})

    def highlight(self, index: int, color: str = "green") -> None:
        self.cmd_queue.put({"cmd": "highlight", "index": index, "color": color})

    def reset_highlight(self) -> None:
        self.cmd_queue.put({"cmd": "reset_highlight"})

    def show_calibration_target(self, index: int) -> None:
        self.cmd_queue.put({"cmd": "show_calibration", "target_index": index})
