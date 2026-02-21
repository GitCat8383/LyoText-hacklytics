"""Pygame-based P300 stimulus flashing window.

Runs as a separate process, communicates with the backend via
multiprocessing queues.
"""

from __future__ import annotations

import random
import time
from multiprocessing import Process, Queue

import config

BG_COLOR = (18, 18, 24)
TILE_COLOR = (35, 39, 52)
TILE_FLASH = (255, 220, 80)
TILE_SELECTED = (80, 200, 120)
TILE_CALIBRATION = (255, 100, 100)
TEXT_COLOR = (220, 220, 230)
TEXT_FLASH = (18, 18, 24)
HEADER_COLOR = (160, 160, 180)
STATUS_COLOR = (100, 100, 120)

WINDOW_W, WINDOW_H = 1280, 800
TILE_MARGIN = 20
COLS, ROWS = 3, 2
HEADER_H = 80
FOOTER_H = 60


def _run_flasher(cmd_queue: Queue, event_queue: Queue) -> None:
    import pygame
    pygame.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Hacklytic - P300 Stimulus")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("Helvetica", 28, bold=True)
    font_header = pygame.font.SysFont("Helvetica", 22)
    font_status = pygame.font.SysFont("Helvetica", 18)

    phrases: list[str] = ["Waiting..."] * config.NUM_PHRASES
    flashing = False
    flash_index = -1
    highlight_index = -1
    highlight_color = TILE_SELECTED
    calibration_target = -1
    status_text = "Waiting for connection..."

    def _tile_rects() -> list:
        usable_w = WINDOW_W - TILE_MARGIN * (COLS + 1)
        usable_h = WINDOW_H - HEADER_H - FOOTER_H - TILE_MARGIN * (ROWS + 1)
        tw = usable_w // COLS
        th = usable_h // ROWS
        rects = []
        for row in range(ROWS):
            for col in range(COLS):
                x = TILE_MARGIN + col * (tw + TILE_MARGIN)
                y = HEADER_H + TILE_MARGIN + row * (th + TILE_MARGIN)
                rects.append(pygame.Rect(x, y, tw, th))
        return rects

    tile_rects = _tile_rects()

    def _draw():
        screen.fill(BG_COLOR)

        header = font_header.render("Focus on your desired phrase", True, HEADER_COLOR)
        screen.blit(header, (WINDOW_W // 2 - header.get_width() // 2, 25))

        for i, rect in enumerate(tile_rects):
            if i >= len(phrases):
                break

            if i == flash_index:
                color = TILE_FLASH
                text_c = TEXT_FLASH
            elif i == highlight_index:
                color = highlight_color
                text_c = TEXT_FLASH
            elif i == calibration_target:
                color = TILE_CALIBRATION
                text_c = TEXT_FLASH
            else:
                color = TILE_COLOR
                text_c = TEXT_COLOR

            pygame.draw.rect(screen, color, rect, border_radius=12)

            text_surf = font_large.render(phrases[i], True, text_c)
            tx = rect.centerx - text_surf.get_width() // 2
            ty = rect.centery - text_surf.get_height() // 2
            screen.blit(text_surf, (tx, ty))

        status_surf = font_status.render(status_text, True, STATUS_COLOR)
        screen.blit(status_surf, (TILE_MARGIN, WINDOW_H - FOOTER_H + 15))

        pygame.display.flip()

    flash_order: list[int] = []
    flash_step = 0
    flash_round = 0
    flash_start_time = 0.0
    in_isi = False

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        while not cmd_queue.empty():
            try:
                cmd = cmd_queue.get_nowait()
            except Exception:
                break

            action = cmd.get("cmd")
            if action == "set_phrases":
                phrases = cmd["phrases"]
                status_text = "Phrases loaded"
            elif action == "start_flash":
                flashing = True
                flash_round = 0
                flash_step = 0
                flash_order = list(range(len(phrases)))
                random.shuffle(flash_order)
                flash_start_time = time.time()
                in_isi = False
                highlight_index = -1
                calibration_target = -1
                status_text = f"Flashing round {flash_round + 1}/{config.NUM_FLASH_ROUNDS}"
            elif action == "stop_flash":
                flashing = False
                flash_index = -1
                status_text = "Flash stopped"
            elif action == "highlight":
                highlight_index = cmd.get("index", -1)
                color_name = cmd.get("color", "green")
                highlight_color = TILE_SELECTED if color_name == "green" else TILE_CALIBRATION
                flash_index = -1
                status_text = f"Selected: {phrases[highlight_index]}" if 0 <= highlight_index < len(phrases) else ""
            elif action == "reset_highlight":
                highlight_index = -1
            elif action == "show_calibration":
                calibration_target = cmd.get("target_index", -1)
                status_text = f"FOCUS on: {phrases[calibration_target]}" if 0 <= calibration_target < len(phrases) else ""
            elif action == "quit":
                running = False

        if flashing and flash_order:
            now = time.time()
            elapsed_ms = (now - flash_start_time) * 1000

            flash_dur = config.FLASH_DURATION_MS
            isi = config.ISI_MS

            if not in_isi:
                if elapsed_ms >= flash_dur:
                    flash_index = -1
                    in_isi = True
                    flash_start_time = now
                elif flash_index == -1:
                    phrase_idx = flash_order[flash_step]
                    flash_index = phrase_idx
                    event_queue.put({
                        "event": "stimulus_onset",
                        "phrase_index": phrase_idx,
                        "timestamp": now,
                    })
            else:
                if elapsed_ms >= isi:
                    in_isi = False
                    flash_step += 1
                    if flash_step >= len(flash_order):
                        flash_step = 0
                        flash_round += 1
                        if flash_round >= config.NUM_FLASH_ROUNDS:
                            flashing = False
                            flash_index = -1
                            event_queue.put({
                                "event": "flash_cycle_complete",
                                "timestamp": now,
                            })
                            status_text = "Classifying..."
                        else:
                            random.shuffle(flash_order)
                            status_text = f"Flashing round {flash_round + 1}/{config.NUM_FLASH_ROUNDS}"
                    flash_start_time = now

        _draw()
        clock.tick(120)

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
