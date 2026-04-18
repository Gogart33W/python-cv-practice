import math
import random
import time
import threading
import json
import os
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree

from state_finder.main import get_state
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info

logger = logging.getLogger("BrawlBot")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

GAME_WIDTH  = 1920
GAME_HEIGHT = 1080

Vec2 = Tuple[float, float]
Box  = Tuple[float, float, float, float]

# ─────────────────────────────────────────────────────────────────────────────
# BRAIN DATA  
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BrainData:
    kiting_multiplier: float = 0.85
    strafe_interval:   float = 1.4
    panic_threshold:   float = 0.42
    lessons_learned:   int   = 0
    games_won:         int   = 0
    games_lost:        int   = 0

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE BRAIN  
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveBrain:
    AMMO_MAX       = 3
    AMMO_REGEN_SEC = 1.7
    SAVE_INTERVAL  = 30.0

    def __init__(self, brawler_name: str) -> None:
        self.brawler    = brawler_name
        os.makedirs("brains", exist_ok=True)
        self._file      = f"brains/brain_{brawler_name}.json"
        self.data       = self._load()
        self._ammo      = self.AMMO_MAX
        self._ammo_lock = threading.Lock()
        self._regen_q:  List[float] = []
        self._last_save = time.time()

    def _load(self) -> BrainData:
        if os.path.exists(self._file):
            try:
                with open(self._file) as f:
                    raw = json.load(f)
                valid = {k: raw[k] for k in BrainData.__dataclass_fields__ if k in raw}
                return BrainData(**valid)
            except Exception: pass
        return BrainData()

    def save(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_save < self.SAVE_INTERVAL:
            return
        try:
            with open(self._file, "w") as f:
                json.dump(self.data.__dict__, f, indent=4)
            self._last_save = now
        except Exception: pass

    @property
    def ammo(self) -> int: return self._ammo

    def consume_ammo(self) -> bool:
        with self._ammo_lock:
            if self._ammo <= 0: return False
            self._ammo -= 1
            self._regen_q.append(time.time())
            return True

    def tick_ammo(self) -> None:
        now = time.time()
        with self._ammo_lock:
            kept = []
            for t in self._regen_q:
                if now - t >= self.AMMO_REGEN_SEC:
                    self._ammo = min(self.AMMO_MAX, self._ammo + 1)
                else: kept.append(t)
            self._regen_q = kept

    def on_game_result(self, rank_or_win) -> None:
        d = self.data
        won = rank_or_win <= 2 if isinstance(rank_or_win, int) else bool(rank_or_win)
        
        if won:
            d.games_won += 1
            d.kiting_multiplier = round(max(0.68, d.kiting_multiplier - 0.008), 4)
        else:
            d.games_lost += 1
            d.kiting_multiplier = round(min(0.96, d.kiting_multiplier + 0.025), 4)
            d.panic_threshold   = round(min(0.60, d.panic_threshold   + 0.015), 4)
            d.strafe_interval   = round(max(0.55, d.strafe_interval   - 0.08),  4)
        d.lessons_learned += 1
        self.save(force=True)

    def punish_too_close(self) -> None:
        d = self.data
        if d.kiting_multiplier < 0.95:
            d.kiting_multiplier = round(d.kiting_multiplier + 0.018, 4)
            d.panic_threshold   = round(min(0.58, d.panic_threshold + 0.008), 4)
            d.strafe_interval   = round(max(0.55, d.strafe_interval - 0.07),  4)
            self.save()

# ─────────────────────────────────────────────────────────────────────────────
# WALL INDEX & TRACKER
# ─────────────────────────────────────────────────────────────────────────────
class WallIndex:
    def __init__(self) -> None:
        self._tree:  Optional[STRtree] = None
        self._polys: List[Polygon]     = []

    def rebuild(self, walls: List[Box]) -> None:
        polys = []
        for x1, y1, x2, y2 in walls:
            x1 += 8; y1 += 8; x2 -= 8; y2 -= 8
            if x2 > x1 and y2 > y1:
                polys.append(Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]))
        self._polys = polys
        self._tree  = STRtree(polys) if polys else None

    def intersects_line(self, line: LineString) -> bool:
        if self._tree is None: return False
        candidates = self._tree.query(line)
        return any(self._polys[i].intersects(line) for i in candidates)

class EnemyTracker:
    WINDOW    = 0.38
    MIN_DT    = 0.06
    MAX_SPEED = 520.0
    MAX_JUMP  = 220.0

    def __init__(self) -> None:
        self._track: List[Tuple[float, Vec2]] = []

    def reset(self) -> None: self._track.clear()

    def push(self, pos: Vec2, t: float) -> None:
        self._track = [(ts, p) for ts, p in self._track if t - ts <= self.WINDOW]
        if self._track:
            last = self._track[-1][1]
            if math.hypot(pos[0]-last[0], pos[1]-last[1]) > self.MAX_JUMP:
                self._track.clear()
        self._track.append((t, pos))

    def velocity(self) -> Optional[Vec2]:
        if len(self._track) < 3: return None
        dt = self._track[-1][0] - self._track[0][0]
        if dt < self.MIN_DT: return None
        vx = (self._track[-1][1][0] - self._track[0][1][0]) / dt
        vy = (self._track[-1][1][1] - self._track[0][1][1]) / dt
        vx = max(-self.MAX_SPEED, min(self.MAX_SPEED, vx))
        vy = max(-self.MAX_SPEED, min(self.MAX_SPEED, vy))
        return (vx, vy)

    def predict(self, lead_time: float) -> Optional[Vec2]:
        vel = self.velocity()
        if vel is None: return None
        px = self._track[-1][1][0] + vel[0] * lead_time
        py = self._track[-1][1][1] + vel[1] * lead_time
        return (px, py)

# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL MOVEMENT (ФІКС ДЖОЙСТИКА)
# ─────────────────────────────────────────────────────────────────────────────
class Movement:
    def __init__(self, window_controller) -> None:
        bot_cfg  = load_toml_as_dict("cfg/bot_config.toml")
        time_cfg = load_toml_as_dict("cfg/time_tresholds.toml")

        self.window_controller = window_controller
        
        self.gamemode_str = str(bot_cfg.get("gamemode", "other")).lower()
        self.gamemode_type = bot_cfg.get("gamemode_type", 3)
        
        self.is_showdown = "sd" in self.gamemode_str or "showdown" in self.gamemode_str
        self.is_brawlball = self.gamemode_type == 3 or "brawl" in self.gamemode_str
        
        self.should_use_gadget = str(bot_cfg.get("bot_uses_gadgets", "no")).lower() in ("yes","true","1")

        self.super_treshold        = time_cfg.get("super", 5.0)
        self.gadget_treshold       = time_cfg.get("gadget", 5.0)
        self.hypercharge_treshold  = time_cfg.get("hypercharge", 5.0)
        
        self._min_move_delay   = float(bot_cfg.get("minimum_movement_delay", 0.1))
        self._unstuck_delay    = float(bot_cfg.get("unstuck_movement_delay", 3.0))
        self._unstuck_duration = float(bot_cfg.get("unstuck_movement_hold_time", 1.5))
        
        self._unstuck_active   = False
        self._unstuck_key      = ""
        self._unstuck_start    = 0.0

        self._keys_held:        List[str] = []
        self._last_movement     = ""
        self._last_movement_time= time.time()
        self._same_move_since   = time.time()
        self._last_move_change_time = 0.0

        self.is_gadget_ready      = False
        self.is_hypercharge_ready = False
        self.is_super_ready       = False
        self.time_since_gadget_checked      = 0.0
        self.time_since_hypercharge_checked = 0.0
        self.time_since_super_checked       = 0.0

        self.TILE_SIZE = 60

    @staticmethod
    def box_center(box: Box) -> Vec2:
        return (box[0]+box[2])/2, (box[1]+box[3])/2

    @staticmethod
    def dist(a: Vec2, b: Vec2) -> float:
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _set_keys(self, keys: List[str]) -> None:
        now = time.time()
        
        # 🔥 ГОЛОВНИЙ ФІКС: Більше ніяких "0.5 сек"! 
        # Якщо ми вже тиснемо правильні кнопки - просто продовжуємо їх тиснути!
        if set(keys) == set(self._keys_held):
            # Щоб емулятор не "заснув" від занадто довгого затискання,
            # перенатискаємо кнопку ТІЛЬКИ якщо пройшло 3 секунди (а не 0.5)
            if now - getattr(self, '_last_refresh_time', 0) > 3.0 and keys:
                self.window_controller.keys_up(self._keys_held)
                time.sleep(0.01)
                self.window_controller.keys_down(keys)
                self._last_refresh_time = now
            return

        down = [k for k in keys if k not in self._keys_held]
        up   = [k for k in self._keys_held if k not in keys]

        if up:   self.window_controller.keys_up(up)
        if down: self.window_controller.keys_down(down)

        self._keys_held = list(keys)
        self._last_movement_time = now
        self._last_refresh_time = now

    def do_movement(self, movement: str) -> None:
        now = time.time()
        movement = "".join(sorted(movement.lower()))

        if movement == self._last_movement:
            if movement and now - self._same_move_since > 3.0:
                movement = random.choice(["w","a","s","d","wa","wd","sa","sd"])
                self._same_move_since = now
        else:
            if now - getattr(self, '_last_move_change_time', 0) < self._min_move_delay and movement != "":
                return
            self._last_move_change_time = now
            self._same_move_since = now
            self._last_movement   = movement

        wanted = [k for k in ["w","a","s","d"] if k in movement]
        self._set_keys(wanted)

    def release_all(self) -> None: 
        self._set_keys([])

    def directed_attack(self, player_pos: Vec2, target_pos: Vec2, attack_range: float, is_super: bool = False, brain: Optional[AdaptiveBrain] = None) -> bool:
        if brain and not is_super and not brain.consume_ammo(): return False
        dx, dy = target_pos[0] - player_pos[0], target_pos[1] - player_pos[1]
        angle, dist = math.atan2(dy, dx), math.hypot(dx, dy)
        jx, jy = (1460, 830) if is_super else (1600, 850)
        jx, jy = int(jx * self.window_controller.width_ratio), int(jy * self.window_controller.height_ratio)
        max_r = 350 * self.window_controller.scale_factor
        pull = min(dist / max(attack_range * 0.8, 1), 1.0)
        r = max(max_r * pull, 60 * self.window_controller.scale_factor)
        end_x, end_y = int(jx + r * math.cos(angle)), int(jy + r * math.sin(angle))
        self.window_controller.swipe(jx, jy, end_x, end_y, duration=0.08)
        time.sleep(0.015)
        return True

    def attack_in_direction(self, direction_str: str) -> None:
        jx, jy = int(1600 * self.window_controller.width_ratio), int(850 * self.window_controller.height_ratio)
        delta = int(200 * self.window_controller.scale_factor)
        dx, dy = 0, 0
        if "w" in direction_str: dy -= delta
        if "s" in direction_str: dy += delta
        if "a" in direction_str: dx -= delta
        if "d" in direction_str: dx += delta
        if dx == 0 and dy == 0:
            dy, dx = (-delta, 0) if self.is_brawlball else (0, delta)
        self.window_controller.swipe(jx, jy, jx+dx, jy+dy, duration=0.08)

    def use_super(self)       -> None: self.window_controller.press_key("E")
    def use_gadget(self)      -> None: self.window_controller.press_key("G")
    def use_hypercharge(self) -> None: self.window_controller.press_key("H")

    def unstuck_movement_if_needed(self, movement: str, now: float) -> str:
        if self._unstuck_active:
            if now - self._unstuck_start > self._unstuck_duration:
                self._unstuck_active = False
            else:
                return self._unstuck_key

        if movement and now - self._same_move_since > self._unstuck_delay + 0.3:
            rev = movement.lower().translate(str.maketrans("wasd","sdwa"))
            if not rev or rev in ("s","w"):
                rev = random.choice(["a","d","wa","wd"])
            self._unstuck_key    = rev
            self._unstuck_active = True
            self._unstuck_start  = now
            return rev
        return movement

# ─────────────────────────────────────────────────────────────────────────────
# PLAY  —  СИНХРОННИЙ І РОЗУМНИЙ БОТ
# ─────────────────────────────────────────────────────────────────────────────
class Play(Movement):

    STATE_GAS   = "GAS!"   
    STATE_HEAL  = "HEAL"
    STATE_FIGHT = "FIGHT"
    STATE_LOOT  = "LOOT"
    STATE_SCOUT = "SCOUT"

    HP_HEAL_ENTER = 35.0   
    HP_HEAL_EXIT  = 75.0   

    def __init__(self, main_info_model, tile_detector_model, window_controller) -> None:
        super().__init__(window_controller)

        bot_cfg  = load_toml_as_dict("cfg/bot_config.toml")
        time_cfg = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main  = Detect(main_info_model,     classes=["enemy","teammate","player","box"])
        self.Detect_tiles = Detect(tile_detector_model, classes=bot_cfg.get("wall_model_classes", []))
        
        self.wall_det_conf   = float(bot_cfg.get("wall_detection_confidence", 0.9))
        self.entity_det_conf = float(bot_cfg.get("entity_detection_confidence", 0.6))

        self.gadget_px_min = float(bot_cfg.get("gadget_pixels_minimum", 2000.0))
        self.hyper_px_min  = float(bot_cfg.get("hypercharge_pixels_minimum", 2000.0))
        self.super_px_min  = float(bot_cfg.get("super_pixels_minimum", 2400.0))

        self.brawlers_info    = load_brawlers_info()
        self._brawler_ranges: Dict[str, Tuple[int,int,int]] = {}

        self._wall_index     = WallIndex()
        self._wall_history:  List[List[Box]] = []
        self._wall_hist_len  = 3
        self._last_wall_proc = 0.0
        self._wall_proc_ivl  = 0.2

        self._ai_lock        = threading.Lock()
        self._current_frame  = None
        self._frame_lock     = threading.Lock()
        self._cached_data:   Optional[dict] = None
        self._cached_game_state = "match"
        self._data_lock      = threading.Lock()
        self._detect_ivl     = 0.035 

        self._tracker        = EnemyTracker()
        self.brain:          Optional[AdaptiveBrain] = None

        self._last_attack_time  = 0.0
        self._attack_cooldown   = 1.1  
        self._last_learn_time   = 0.0
        self._last_loop_time    = 0.0
        self.frame_time         = 1 / 60

        self._real_hp           = 100.0
        self._last_hp_check     = 0.0
        self._hp_check_interval = 0.18  

        self._human_state   = self.STATE_SCOUT
        self._explore_dir   = ""
        self._explore_time  = 0.0
        self._strafe_flip   = 0.0
        self._strafe_sign   = 1.0

        self._last_frame_hash = 0
        self._freeze_start    = time.time()
        self._blind_dir       = ""
        self._blind_dir_time  = 0.0
        self._last_det: Dict[str, float] = {"player": time.time(), "enemy": time.time()}

        self.scene_data: List[dict] = []

        self._running       = True
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

    def _vision_loop(self) -> None:
        while self._running:
            with self._frame_lock: frame = self._current_frame
            if frame is not None:
                try:
                    with self._ai_lock:
                        result = self.Detect_main.detect_objects(frame, conf_tresh=self.entity_det_conf)
                        if int(time.time() * 10) % 5 == 0:
                            tiles = self.Detect_tiles.detect_objects(frame, conf_tresh=self.wall_det_conf)
                            walls = [b for cls, boxes in tiles.items() if cls != "bush" for b in boxes]
                            self._wall_history.append(walls)
                            if len(self._wall_history) > self._wall_hist_len: self._wall_history.pop(0)
                            merged = {}
                            for wlist in self._wall_history:
                                for w in wlist: merged[tuple(w)] = merged.get(tuple(w), 0) + 1
                            self._wall_index.rebuild([list(k) for k, cnt in merged.items() if cnt >= 1])
                        g_state = get_state(frame)
                    with self._data_lock:
                        self._cached_data = result
                        self._cached_game_state = g_state
                except Exception: pass
            time.sleep(self._detect_ivl)

    def _get_gas_vector(self, frame, player_pos: Vec2) -> Tuple[float, float]:
        if frame is None or not self.is_showdown: return 0.0, 0.0
        try:
            arr = np.array(frame) if isinstance(frame, Image.Image) else frame
            small = cv2.resize(arr, (GAME_WIDTH // 10, GAME_HEIGHT // 10))
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            
            lower_green = np.array([45, 150, 150])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            y_coords, x_coords = np.where(mask > 0)
            
            if len(x_coords) < 10: return 0.0, 0.0 
                
            px, py = player_pos[0] / 10.0, player_pos[1] / 10.0
            gx, gy = 0.0, 0.0
            fear_radius = 45.0 * self.window_controller.scale_factor
            
            for x, y in zip(x_coords, y_coords):
                dist = math.hypot(x - px, y - py)
                if dist < fear_radius:
                    weight = 1.0 / max(dist, 1.0)
                    gx += (px - x) * weight 
                    gy += (py - y) * weight
                    
            length = math.hypot(gx, gy)
            if length > 0:
                return (gx / length) * 500.0, (gy / length) * 500.0
        except Exception: pass
        return 0.0, 0.0

    def _read_real_hp(self, frame, now: float) -> float:
        if now - self._last_hp_check < self._hp_check_interval: return self._real_hp
        self._last_hp_check = now
        try:
            w = self.window_controller
            crop = frame.crop((
                int(55  * w.width_ratio),  int(1018 * w.height_ratio),
                int(310 * w.width_ratio),  int(1042 * w.height_ratio),
            ))
            green_px = count_hsv_pixels(crop, (42, 90, 90),  (90, 255, 255))
            total_px = count_hsv_pixels(crop, (0,  0,  30),  (180,255,255))
            if total_px > 10: self._real_hp = max(0.0, min(100.0, green_px / total_px * 100.0))
        except Exception: pass
        return self._real_hp

    def _check_abilities(self, frame, now: float) -> None:
        w = self.window_controller
        for name, th, pxmin, color in [
            ("hypercharge", self.hypercharge_treshold, self.hyper_px_min, [(137,158,159),(179,255,255)]),
            ("gadget", self.gadget_treshold, self.gadget_px_min, [(57,219,165),(62,255,255)]),
            ("super", self.super_treshold, self.super_px_min, [(17,170,200),(27,255,255)])
        ]:
            if now - getattr(self, f"time_since_{name}_checked") > th:
                try:
                    c = (1350,940,1450,1050) if name=="hypercharge" else (1580,930,1700,1050) if name=="gadget" else (1460,830,1560,930)
                    crop = frame.crop((int(c[0]*w.width_ratio), int(c[1]*w.height_ratio), int(c[2]*w.width_ratio), int(c[3]*w.height_ratio)))
                    setattr(self, f"is_{name}_ready", count_hsv_pixels(crop, color[0], color[1]) > pxmin)
                except Exception: pass
                setattr(self, f"time_since_{name}_checked", now)

    def _get_range(self, brawler: str) -> Tuple[int,int,int]:
        if brawler not in self._brawler_ranges:
            sf   = self.window_controller.scale_factor
            info = self.brawlers_info.get(brawler, {})
            self._brawler_ranges[brawler] = (
                int(info.get("safe_range",   400) * sf),
                int(info.get("attack_range", 600) * sf),
                int(info.get("super_range",  600) * sf),
            )
        return self._brawler_ranges[brawler]

    def _can_pierce_walls(self, brawler: str, skill: str) -> bool:
        key = "ignore_walls_for_attacks" if skill == "attack" else "ignore_walls_for_supers"
        return bool(self.brawlers_info.get(brawler, {}).get(key, False))

    def _has_los(self, a: Vec2, b: Vec2) -> bool: return not self._wall_index.intersects_line(LineString([a, b]))

    def _is_path_clear(self, pos: Vec2, move: str) -> bool:
        d  = self.TILE_SIZE * self.window_controller.scale_factor
        dx = ("d" in move) * d - ("a" in move) * d
        dy = ("s" in move) * d - ("w" in move) * d
        if dx == 0 and dy == 0: return True
        return not self._wall_index.intersects_line(LineString([pos, (pos[0]+dx, pos[1]+dy)]))

    def _best_free_move(self, pos: Vec2, preferred: str) -> str:
        candidates = [preferred] + [m for m in ["w","a","s","d","wa","wd","sa","sd"] if m != preferred]
        for m in candidates:
            if m and self._is_path_clear(pos, m): return m
        return preferred

    def _find_best_target(self, entities: List[Box], player_pos: Vec2, skill: str) -> Optional[Tuple[Vec2, float]]:
        pierce = self._can_pierce_walls(self.brain.brawler if self.brain else "", skill)
        best_pos, best_dist = None, float("inf")
        for box in entities:
            ep, d = self.box_center(box), self.dist(self.box_center(box), player_pos)
            if (pierce or self._has_los(player_pos, ep)) and d < best_dist: best_pos, best_dist = ep, d
        if not best_pos:
            for box in entities:
                ep, d = self.box_center(box), self.dist(self.box_center(box), player_pos)
                if d < best_dist: best_pos, best_dist = ep, d
        return (best_pos, best_dist) if best_pos else None

    def _handle_attack(self, player_pos: Vec2, enemy_pos: Vec2, enemy_dist: float, attack_range: float, super_range: float, brawler_info: dict, now: float) -> None:
        brawler = self.brain.brawler if self.brain else ""
        has_los_atk = self._can_pierce_walls(brawler, "attack") or self._has_los(player_pos, enemy_pos)

        if self._human_state == self.STATE_HEAL and enemy_dist > attack_range * 0.45: return

        if enemy_dist <= attack_range and has_los_atk and (self.brain.ammo > 0 if self.brain else True) and (now - self._last_attack_time > self._attack_cooldown * 0.85):
            if self.should_use_gadget and self.is_gadget_ready:
                self.use_gadget()
                self.is_gadget_ready = False
            if self.is_hypercharge_ready:
                self.use_hypercharge()
                self.is_hypercharge_ready = False

            self._tracker.push(enemy_pos, now)
            lead = min(enemy_dist / max(float(brawler_info.get("projectile_speed", 1100)), 1), 0.55)
            pred = self._tracker.predict(lead) or enemy_pos
            if self.directed_attack(player_pos, pred, attack_range, is_super=False, brain=self.brain):
                self._last_attack_time = now

        if not self.is_super_ready: return
        if (self._can_pierce_walls(brawler, "super") or self._has_los(player_pos, enemy_pos)) and (enemy_dist <= super_range or brawler_info.get("super_type") in ("spawnable","other")):
            self._tracker.push(enemy_pos, now)
            pred = self._tracker.predict(min(enemy_dist / max(float(brawler_info.get("projectile_speed", 1100)), 1), 0.55)) or enemy_pos
            if self.directed_attack(player_pos, pred, attack_range, is_super=True):
                self.is_super_ready = False

    def _get_movement(self, data: dict, brawler: str) -> str:
        brawler_info = self.brawlers_info.get(brawler, {})
        self._attack_cooldown = float(brawler_info.get("attack_cooldown", 1.1))

        _, attack_range, super_range = self._get_range(brawler)
        player_pos = self.box_center(data["player"][0])
        now        = time.time()

        enemies   = data.get("enemy")    or []
        boxes     = data.get("box")      or []
        teammates = data.get("teammate") or []

        real_hp = self._read_real_hp(data.get("_frame"), now)
        gas_vx, gas_vy = self._get_gas_vector(data.get("_frame"), player_pos)
        is_gas_near = math.hypot(gas_vx, gas_vy) > 0

        # ── FSM TRANSITIONS ──
        if is_gas_near:
            self._human_state = self.STATE_GAS
        elif real_hp <= self.HP_HEAL_ENTER:
            self._human_state = self.STATE_HEAL
        elif real_hp >= self.HP_HEAL_EXIT and self._human_state == self.STATE_HEAL:
            self._human_state = self.STATE_SCOUT

        if self._human_state not in (self.STATE_HEAL, self.STATE_GAS):
            enemy_target = self._find_best_target(enemies, player_pos, "attack")
            box_target   = self._find_best_target(boxes, player_pos, "attack") if self.is_showdown else None

            if enemy_target and box_target:
                if enemy_target[1] < box_target[1] * 1.5 or enemy_target[1] < attack_range:
                    self._human_state = self.STATE_FIGHT
                else:
                    self._human_state = self.STATE_LOOT
            elif enemy_target: self._human_state = self.STATE_FIGHT
            elif box_target:   self._human_state = self.STATE_LOOT
            else:              self._human_state = self.STATE_SCOUT

        # ── STATE: GAS! ──
        if self._human_state == self.STATE_GAS:
            movement = ("d" if gas_vx > 0 else "a") + ("s" if gas_vy > 0 else "w")
            target = self._find_best_target(enemies, player_pos, "attack")
            if target: 
                self._handle_attack(player_pos, target[0], target[1], attack_range, super_range, brawler_info, now)
            return self._best_free_move(player_pos, movement)

        # ── STATE: HEAL ──
        if self._human_state == self.STATE_HEAL:
            if enemies:
                ex, ey = self.box_center(min(enemies, key=lambda b: self.dist(self.box_center(b), player_pos)))
                movement = ("d" if player_pos[0]-ex > 0 else "a") + ("s" if player_pos[1]-ey > 0 else "w")
                target = self._find_best_target(enemies, player_pos, "attack")
                if target: self._handle_attack(player_pos, target[0], target[1], attack_range, super_range, brawler_info, now)
            elif teammates and not self.is_showdown:
                tx, ty = self.box_center(teammates[0])
                movement = ("d" if tx-player_pos[0] > 0 else "a") + ("s" if ty-player_pos[1] > 0 else "w")
            else:
                movement = "s" if self.is_brawlball else random.choice(["a", "s"])
            return self._best_free_move(player_pos, movement)

        # ── STATE: SCOUT ──
        if self._human_state == self.STATE_SCOUT:
            self._tracker.reset()
            if not self._explore_dir or now - self._explore_time > 2.0 or not self._is_path_clear(player_pos, self._explore_dir):
                if self.is_brawlball:
                    bias = "w"
                elif self.is_showdown:
                    bias = random.choice(["w", "a", "s", "d", "wa", "wd", "sa", "sd"]) 
                else:
                    bias = random.choice(["w", "wa", "wd"]) 
                self._explore_dir  = bias
                self._explore_time = now
            return self._best_free_move(player_pos, self._explore_dir)

        # ── STATE: LOOT ──
        if self._human_state == self.STATE_LOOT:
            target = self._find_best_target(boxes, player_pos, "attack")
            if target is None:
                self._human_state = self.STATE_SCOUT
                return self._best_free_move(player_pos, self._explore_dir)
            box_pos, box_dist = target
            
            if box_dist > attack_range * 0.75:
                movement = ("d" if box_pos[0]-player_pos[0] > 0 else "a") + ("s" if box_pos[1]-player_pos[1] > 0 else "w")
            else:
                movement = "" 
                
            self._handle_attack(player_pos, box_pos, box_dist, attack_range, super_range, brawler_info, now)
            return self._best_free_move(player_pos, movement)

        # ── STATE: FIGHT ──
        target = self._find_best_target(enemies, player_pos, "attack")
        if target is None:
            self._human_state = self.STATE_SCOUT
            return self._best_free_move(player_pos, self._explore_dir)

        enemy_pos, enemy_dist = target

        if enemy_dist < attack_range * 0.35 and self.brain:
            if now - self._last_learn_time > 6.0:
                self.brain.punish_too_close()
                self._last_learn_time = now

        kiting_mul = self.brain.data.kiting_multiplier if self.brain else 0.85
        ideal_dist = attack_range * kiting_mul
        panic_thr  = self.brain.data.panic_threshold  if self.brain else 0.42
        strafe_ivl = self.brain.data.strafe_interval  if self.brain else 1.4

        if now - self._strafe_flip > strafe_ivl:
            self._strafe_sign *= -1
            self._strafe_flip  = now

        walls_block = not self._has_los(player_pos, enemy_pos)
        dx, dy = enemy_pos[0] - player_pos[0], enemy_pos[1] - player_pos[1]

        if enemy_dist < attack_range * panic_thr:
            movement = ("d" if dx < 0 else "a") + ("s" if dy < 0 else "w")
        elif walls_block and not brawler_info.get("ignore_walls_for_attacks", False):
            vx, vy = dx * 0.4 + dy * self._strafe_sign * 1.6, dy * 0.4 - dx * self._strafe_sign * 1.6
            movement = ("d" if vx > 0 else "a") + ("s" if vy > 0 else "w")
        elif enemy_dist < ideal_dist:
            px, py = -dy * self._strafe_sign * 0.6, dx * self._strafe_sign * 0.6
            movement  = ("d" if -dx + px > 0 else "a") + ("s" if -dy + py > 0 else "w")
        elif enemy_dist <= attack_range:
            movement = ("d" if -dy * self._strafe_sign > 0 else "a") + ("s" if dx * self._strafe_sign > 0 else "w")
        else:
            px, py = dx * 0.75 - dy * self._strafe_sign * 0.4, dy * 0.75 + dx * self._strafe_sign * 0.4
            movement = ("d" if px > 0 else "a") + ("s" if py > 0 else "w")

        self._handle_attack(player_pos, enemy_pos, enemy_dist, attack_range, super_range, brawler_info, now)
        return self._best_free_move(player_pos, movement)

    def _frame_hash(self, frame) -> int:
        if frame is None: return 0
        try: return int(np.sum((np.array(frame) if isinstance(frame, Image.Image) else frame)[::10, ::10, 0]) % 2**31)
        except Exception: return 0

    @staticmethod
    def _validate(data: dict) -> Optional[dict]:
        if "player" not in data or not data["player"]: return None
        data.setdefault("enemy",    None)
        data.setdefault("teammate", [])
        data.setdefault("box",      [])
        data.setdefault("wall",     [])
        return data

    def main(self, frame, brawler: str) -> None:
        now = time.time()
        with self._frame_lock: self._current_frame = frame

        h = self._frame_hash(frame)
        if h != self._last_frame_hash: self._freeze_start = now
        self._last_frame_hash = h
        is_frozen = (now - self._freeze_start) > 1.2

        if self.brain is None and brawler: self.brain = AdaptiveBrain(brawler)
        if self.brain: self.brain.tick_ammo()
        self._check_abilities(frame, now)

        with self._data_lock:
            data = dict(self._cached_data) if self._cached_data else {}
            g_state = getattr(self, "_cached_game_state", "match")

        if data.get("player"): self._last_det["player"] = now
        if data.get("enemy"): self._last_det["enemy"] = now

        # АНТИ-АФК
        if is_frozen or not data or not data.get("player"):
            if g_state == "match":
                if now - self._blind_dir_time > 1.5:
                    if self.is_brawlball:
                        opts = ["w", "wa", "wd"]
                    elif self.is_showdown:
                        opts = ["w", "wa", "wd", "a", "d", "s", "sa", "sd"]
                    else:
                        opts = ["w", "wa", "wd", "a", "d"]
                    self._blind_dir = random.choice(opts)
                    self._blind_dir_time = now

                self.do_movement(self._blind_dir)
                if now - self._last_attack_time > 2.0:
                    self.attack_in_direction(self._blind_dir)
                    self._last_attack_time = now
            else: self.release_all()
            return

        valid_data = self._validate(data)
        if valid_data is None: return
        valid_data["_frame"] = frame

        movement = self._get_movement(valid_data, brawler)
        movement = self.unstuck_movement_if_needed(movement, now)
        self.do_movement(movement)

        self.scene_data.append({
            "frame":  len(self.scene_data),
            "player": valid_data.get("player", []),
            "enemy":  valid_data.get("enemy",  []),
            "wall":   valid_data.get("wall",   []),
            "move":   movement,
            "hp":     self._real_hp,
            "state":  self._human_state,
        })
        if len(self.scene_data) > 600: self.scene_data.pop(0)

    def on_game_result(self, rank: int) -> None:
        if self.brain: self.brain.on_game_result(rank)

    def generate_visualization(self, output: str = "visualization.mp4") -> None:
        pass