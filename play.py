import math
import random
import time
import threading
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from shapely import LineString
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from state_finder.main import get_state
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info

logger = logging.getLogger("BrawlBot")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

GAME_WIDTH  = 1920
GAME_HEIGHT = 1080

# ──────────────────────────────────────────────
# ТИПИ
# ──────────────────────────────────────────────
Vec2 = tuple[float, float]
Box  = tuple[float, float, float, float]  # x1 y1 x2 y2

@dataclass
class BrainData:
    kiting_multiplier: float = 0.85
    prediction_lead:   float = 0.85
    strafe_interval:   float = 1.5
    panic_threshold:   float = 0.45
    lessons_learned:   int   = 0

# ──────────────────────────────────────────────
# ADAPTIVE BRAIN
# ──────────────────────────────────────────────
class AdaptiveBrain:
    AMMO_MAX       = 3
    AMMO_REGEN_SEC = 1.7
    SAVE_INTERVAL  = 30.0

    def __init__(self, brawler_name: str) -> None:
        self.brawler   = brawler_name
        os.makedirs("brains", exist_ok=True)
        self._file     = f"brains/brain_{brawler_name}.json"
        self.data      = self._load()
        self._ammo     = self.AMMO_MAX
        self._ammo_lock = threading.Lock()
        self._regen_start: list[float] = []
        self._last_save  = time.time()

    def _load(self) -> BrainData:
        if os.path.exists(self._file):
            try:
                with open(self._file) as f:
                    return BrainData(**json.load(f))
            except Exception as e:
                logger.warning("Brain load failed (%s), using defaults.", e)
        return BrainData()

    def save(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_save < self.SAVE_INTERVAL:
            return
        try:
            with open(self._file, "w") as f:
                json.dump(self.data.__dict__, f, indent=4)
            self._last_save = now
        except Exception as e:
            logger.warning("Brain save failed: %s", e)

    @property
    def ammo(self) -> int:
        return self._ammo

    def consume_ammo(self) -> bool:
        with self._ammo_lock:
            if self._ammo <= 0:
                return False
            self._ammo -= 1
            self._regen_start.append(time.time())
            return True

    def tick_ammo(self) -> None:
        now = time.time()
        with self._ammo_lock:
            kept = []
            for t in self._regen_start:
                if now - t >= self.AMMO_REGEN_SEC:
                    self._ammo = min(self.AMMO_MAX, self._ammo + 1)
                else:
                    kept.append(t)
            self._regen_start = kept

    def punish_too_close(self) -> None:
        if self.data.kiting_multiplier < 0.95:
            self.data.kiting_multiplier = round(self.data.kiting_multiplier + 0.02, 4)
            self.data.panic_threshold = round(min(0.60, self.data.panic_threshold + 0.01), 4)
            self.data.strafe_interval = round(max(0.6, self.data.strafe_interval - 0.1), 4)
            self.data.lessons_learned  += 1
            self.save()

    def reward_safe_attack(self) -> None:
        if self.data.kiting_multiplier > 0.70:
            self.data.kiting_multiplier = round(self.data.kiting_multiplier - 0.005, 4)
            self.save()

# ──────────────────────────────────────────────
# WALL SPATIAL INDEX
# ──────────────────────────────────────────────
class WallIndex:
    def __init__(self) -> None:
        self._tree: Optional[STRtree] = None
        self._polys: list[Polygon]    = []

    def rebuild(self, walls: list[Box]) -> None:
        self._polys = []
        for x1, y1, x2, y2 in walls:
            x1 += 10; y1 += 10; x2 -= 10; y2 -= 10
            if x2 > x1 and y2 > y1:
                self._polys.append(Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)]))
        self._tree = STRtree(self._polys) if self._polys else None

    def intersects_line(self, line: LineString) -> bool:
        if self._tree is None:
            return False
        candidates = self._tree.query(line)
        return any(self._polys[i].intersects(line) for i in candidates)

# ──────────────────────────────────────────────
# ENEMY VELOCITY TRACKER
# ──────────────────────────────────────────────
class EnemyTracker:
    WINDOW   = 0.35
    MIN_DT   = 0.08
    MAX_SPEED = 500.0
    MAX_JUMP  = 200.0

    def __init__(self) -> None:
        self._track: list[tuple[float, Vec2]] = []

    def reset(self) -> None:
        self._track.clear()

    def push(self, pos: Vec2, t: float) -> None:
        self._track = [p for p in self._track if t - p[0] <= self.WINDOW]
        if self._track:
            last_pos = self._track[-1][1]
            dist = math.hypot(pos[0]-last_pos[0], pos[1]-last_pos[1])
            if dist > self.MAX_JUMP:
                self._track.clear()
        self._track.append((t, pos))

    def predict(self, lead_time: float) -> Optional[Vec2]:
        if len(self._track) < 3:
            return None
        dt = self._track[-1][0] - self._track[0][0]
        if dt < self.MIN_DT:
            return None
        vx = (self._track[-1][1][0] - self._track[0][1][0]) / dt
        vy = (self._track[-1][1][1] - self._track[0][1][1]) / dt
        vx = max(-self.MAX_SPEED, min(self.MAX_SPEED, vx))
        vy = max(-self.MAX_SPEED, min(self.MAX_SPEED, vy))
        px = self._track[-1][1][0] + vx * lead_time
        py = self._track[-1][1][1] + vy * lead_time
        return (px, py)

# ──────────────────────────────────────────────
# LOW-LEVEL MOVEMENT
# ──────────────────────────────────────────────
class Movement:
    def __init__(self, window_controller) -> None:
        bot_cfg  = load_toml_as_dict("cfg/bot_config.toml")
        time_cfg = load_toml_as_dict("cfg/time_tresholds.toml")

        self.window_controller = window_controller
        self.game_mode         = bot_cfg["gamemode_type"]
        self.should_use_gadget = str(bot_cfg.get("bot_uses_gadgets","no")).lower() in ("yes","true","1")

        self.super_treshold        = time_cfg["super"]
        self.gadget_treshold       = time_cfg["gadget"]
        self.hypercharge_treshold  = time_cfg["hypercharge"]
        self.walls_treshold        = time_cfg["wall_detection"]
        self.keep_walls_in_memory  = self.walls_treshold <= 1

        self._unstuck_delay    = bot_cfg["unstuck_movement_delay"]
        self._unstuck_duration = bot_cfg["unstuck_movement_hold_time"]
        self._unstuck_active   = False
        self._unstuck_key      = ""
        self._unstuck_start    = 0.0

        self._keys_held: list[str] = []
        self._last_movement        = ""
        self._last_movement_time   = time.time()
        self._same_move_since      = time.time()

        self.is_gadget_ready      = False
        self.is_hypercharge_ready = False
        self.is_super_ready       = False

        self.time_since_gadget_checked     = 0.0
        self.time_since_hypercharge_checked= 0.0
        self.time_since_super_checked      = 0.0

        self.TILE_SIZE = 60

    @staticmethod
    def box_center(box: Box) -> Vec2:
        return (box[0]+box[2])/2, (box[1]+box[3])/2

    @staticmethod
    def dist(a: Vec2, b: Vec2) -> float:
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _set_keys(self, keys: list[str]) -> None:
        now = time.time()

        # 🔥 ФІКС №2: Force Refresh (відлипання кнопок в емуляторі)
        if set(keys) == set(self._keys_held):
            if now - getattr(self, "_last_movement_time", 0) < 0.4:
                return
            # Примусово відпускаємо і чекаємо мілісекунду, щоб емулятор зрозумів
            self.window_controller.keys_up(self._keys_held)
            time.sleep(0.01)
            self._keys_held = []

        down = [k for k in keys if k not in self._keys_held]
        up   = [k for k in self._keys_held if k not in keys]

        if up:
            self.window_controller.keys_up(up)
        if down:
            self.window_controller.keys_down(down)

        self._keys_held = keys
        self._last_movement_time = now

    def do_movement(self, movement: str) -> None:
        now = time.time()
        
        if movement == self._last_movement:
            if movement != "" and now - self._same_move_since > 3.0:
                movement = random.choice(["w", "a", "s", "d", "wa", "wd", "sa", "sd"])
                self._same_move_since = now
        else:
            self._same_move_since = now
            self._last_movement = movement

        wanted = [k for k in ["w","a","s","d"] if k in movement.lower()]
        self._set_keys(wanted)

    def release_all(self) -> None:
        self._set_keys([])

    def directed_attack(self, player_pos: Vec2, target_pos: Vec2,
                         attack_range: float, is_super: bool = False,
                         brain: Optional[AdaptiveBrain] = None) -> bool:
        if brain and not is_super and not brain.consume_ammo():
            return False

        dx = target_pos[0] - player_pos[0]
        dy = target_pos[1] - player_pos[1]
        angle = math.atan2(dy, dx)
        dist  = math.hypot(dx, dy)

        if is_super:
            jx = int(1460 * self.window_controller.width_ratio)
            jy = int(830  * self.window_controller.height_ratio)
        else:
            jx = int(1600 * self.window_controller.width_ratio)
            jy = int(850  * self.window_controller.height_ratio)

        max_r     = 350 * self.window_controller.scale_factor
        pull      = min(dist / max(attack_range * 0.8, 1), 1.0)
        r         = max(max_r * pull, 60 * self.window_controller.scale_factor)
        end_x     = int(jx + r * math.cos(angle))
        end_y     = int(jy + r * math.sin(angle))

        self.window_controller.swipe(jx, jy, end_x, end_y, duration=0.12)
        time.sleep(0.02)
        return True
        
    def attack_in_direction(self, direction_str: str) -> None:
        jx = int(1600 * self.window_controller.width_ratio)
        jy = int(850  * self.window_controller.height_ratio)
        
        delta = int(200 * self.window_controller.scale_factor)
        dx, dy = 0, 0
        
        if "w" in direction_str: dy -= delta
        if "s" in direction_str: dy += delta
        if "a" in direction_str: dx -= delta
        if "d" in direction_str: dx += delta
        
        if dx == 0 and dy == 0:
            if self.game_mode == 3: 
                dy -= delta 
            else:
                dx += delta
                
        end_x = jx + dx
        end_y = jy + dy
        
        self.window_controller.swipe(jx, jy, end_x, end_y, duration=0.1)

    def use_super(self)       -> None: self.window_controller.press_key("E")
    def use_gadget(self)      -> None: self.window_controller.press_key("G")
    def use_hypercharge(self) -> None: self.window_controller.press_key("H")

    def unstuck_movement_if_needed(self, movement: str, now: float) -> str:
        if self._unstuck_active:
            if now - self._unstuck_start > self._unstuck_duration:
                self._unstuck_active = False
            else:
                return self._unstuck_key

        if movement != "" and now - self._same_move_since > self._unstuck_delay + 0.3:
            rev = movement.lower().translate(str.maketrans("wasd","sdwa"))
            if not rev or rev in ("s","w"):
                rev = random.choice(["a", "d", "wa", "wd"])
            self._unstuck_key    = rev
            self._unstuck_active = True
            self._unstuck_start  = now
            return rev

        return movement

# ──────────────────────────────────────────────
# PLAY — головна логіка
# ──────────────────────────────────────────────
class Play(Movement):
    def __init__(self, main_info_model, tile_detector_model, window_controller) -> None:
        super().__init__(window_controller)

        bot_cfg  = load_toml_as_dict("cfg/bot_config.toml")
        time_cfg = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main      = Detect(main_info_model,     classes=["enemy","teammate","player"])
        self.Detect_tiles     = Detect(tile_detector_model, classes=bot_cfg["wall_model_classes"])
        self.wall_det_conf    = bot_cfg["wall_detection_confidence"]
        self.entity_det_conf  = bot_cfg["entity_detection_confidence"]

        self.brawlers_info    = load_brawlers_info()
        self._brawler_ranges: dict = {}

        self.no_detect_delay  = time_cfg["no_detection_proceed"]
        self.min_move_delay   = bot_cfg["minimum_movement_delay"]
        self.gadget_px_min    = bot_cfg["gadget_pixels_minimum"]
        self.hyper_px_min     = bot_cfg["hypercharge_pixels_minimum"]
        self.super_px_min     = bot_cfg["super_pixels_minimum"]

        self.should_detect_walls = bot_cfg.get("gamemode","") in ("brawlball","brawl_ball","brawll ball")

        self._wall_index      = WallIndex()
        self._wall_history:   list[list[Box]] = []
        self._wall_hist_len   = 3
        self._last_walls:     list[Box] = []
        self._last_wall_proc  = 0.0
        self._wall_proc_ivl   = 0.10

        self._current_frame   = None
        self._frame_lock      = threading.Lock()
        self._cached_data: Optional[dict] = None
        self._data_lock       = threading.Lock()
        self._detect_ivl      = 0.03

        self._tracker         = EnemyTracker()
        self.brain: Optional[AdaptiveBrain] = None

        self._last_attack_time  = 0.0
        self._attack_cooldown   = 1.0 
        self._last_learn_time   = 0.0
        self._last_no_det_proc  = 0.0
        self._last_loop_time    = 0.0
        self.frame_time         = 1 / 60

        self._last_hit_taken    = 0.0
        self._last_det: dict[str, float] = {"player": time.time(), "enemy": time.time()}
        self.scene_data: list[dict] = []

        self._last_state_check_time = 0.0
        self._cached_state = ""
        self._last_blind_move_time = 0.0
        self._blind_move_dir = ""
        
        # Для Anti-Freeze
        self._last_frame_hash = None
        self._freeze_start = time.time()

        self._running = True
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self._vision_thread.start()

    @property
    def time_since_detections(self) -> dict[str, float]:
        return self._last_det

    def _vision_loop(self) -> None:
        while self._running:
            with self._frame_lock:
                frame = self._current_frame
            if frame is not None:
                try:
                    result = self.Detect_main.detect_objects(frame, conf_tresh=self.entity_det_conf)
                    with self._data_lock:
                        self._cached_data = result
                except Exception as e:
                    logger.debug("Vision error: %s", e)
            time.sleep(self._detect_ivl)

    def stop(self) -> None:
        self._running = False

    def _get_range(self, brawler: str) -> tuple[int,int,int]:
        if brawler not in self._brawler_ranges:
            sf = self.window_controller.scale_factor
            info = self.brawlers_info[brawler]
            self._brawler_ranges[brawler] = (
                int(info["safe_range"]   * sf),
                int(info["attack_range"] * sf),
                int(info["super_range"]  * sf),
            )
        return self._brawler_ranges[brawler]

    def _can_pierce_walls(self, brawler: str, skill: str) -> bool:
        key = "ignore_walls_for_attacks" if skill=="attack" else "ignore_walls_for_supers"
        return bool(self.brawlers_info.get(brawler,{}).get(key, False))

    def _has_los(self, a: Vec2, b: Vec2) -> bool:
        return not self._wall_index.intersects_line(LineString([a, b]))

    def _is_path_clear(self, pos: Vec2, move: str, distance: Optional[float]=None) -> bool:
        d  = distance or self.TILE_SIZE * self.window_controller.scale_factor
        dx = ("d" in move) * d - ("a" in move) * d
        dy = ("s" in move) * d - ("w" in move) * d
        return not self._wall_index.intersects_line(LineString([pos, (pos[0]+dx, pos[1]+dy)]))

    def _best_free_move(self, pos: Vec2, preferred: str, exclude: str="") -> str:
        candidates = [preferred] + [m for m in ["w","a","s","d", "wa", "wd", "sa", "sd"] if m!=exclude and m!=preferred]
        for m in candidates:
            if m and self._is_path_clear(pos, m):
                return m
        return preferred

    def _find_best_target(self, enemies: list[Box], player_pos: Vec2, skill: str) -> Optional[tuple[Vec2,float]]:
        pierce = self._can_pierce_walls(self.brain.brawler if self.brain else "", skill)
        best_pos, best_dist = None, float("inf")
        for box in enemies:
            ep   = self.box_center(box)
            d    = self.dist(ep, player_pos)
            los  = pierce or self._has_los(player_pos, ep)
            if los and d < best_dist:
                best_pos, best_dist = ep, d
        if best_pos is None:
            for box in enemies:
                ep = self.box_center(box)
                d  = self.dist(ep, player_pos)
                if d < best_dist:
                    best_pos, best_dist = ep, d
        return (best_pos, best_dist) if best_pos else None

    def _calc_kite_vector(self, player_pos: Vec2, enemy_pos: Vec2,
                           enemy_dist: float, ideal_dist: float,
                           panic_target: Optional[Vec2]) -> Vec2:
        if panic_target:
            return (panic_target[0]-player_pos[0], panic_target[1]-player_pos[1])

        dx = enemy_pos[0] - player_pos[0]
        dy = enemy_pos[1] - player_pos[1]

        err    = enemy_dist - ideal_dist
        scale  = err / max(enemy_dist, 1)
        rx, ry = dx*scale, dy*scale

        strafe_time = self.brain.data.strafe_interval if self.brain else 1.5
        sf     = 1 if int(time.time()/strafe_time) % 2 == 0 else -1
        px, py = dy*sf*0.8, -dx*sf*0.8

        return (rx+px, ry+py)

    def _vector_to_keys(self, vx: float, vy: float, threshold: float=1.5) -> str:
        h = "d" if vx > threshold else "a" if vx < -threshold else ""
        v = "s" if vy > threshold else "w" if vy < -threshold else ""
        return h+v

    def _get_panic_target(self, player_pos: Vec2, enemy_pos: Vec2,
                           enemy_dist: float, attack_range: float,
                           teammates: list[Box]) -> Optional[Vec2]:
        is_regen   = time.time() - self._last_hit_taken < 3.0
        
        panic_thresh = self.brain.data.panic_threshold if self.brain else 0.45
        is_pressed = enemy_dist < attack_range * panic_thresh

        if not is_regen and not is_pressed:
            return None
        if is_pressed:
            self._last_hit_taken = time.time()

        if teammates:
            tm = self.box_center(teammates[0])
            d_et = self.dist(enemy_pos, tm)
            if d_et > 1:
                ux  = (tm[0]-enemy_pos[0]) / d_et
                uy  = (tm[1]-enemy_pos[1]) / d_et
                return (tm[0]+ux*80, tm[1]+uy*80)

        dx_away = enemy_pos[0] - player_pos[0]
        dy_away = enemy_pos[1] - player_pos[1]
        return (player_pos[0] - dx_away * 3, player_pos[1] - dy_away * 3)

    def _predicted_pos(self, enemy_pos: Vec2, player_pos: Vec2,
                        attack_range: float, now: float) -> Vec2:
        self._tracker.push(enemy_pos, now)
        pred = self._tracker.predict(0.25)
        if pred:
            return pred
        return enemy_pos

    def _handle_attack(self, player_pos: Vec2, enemy_pos: Vec2,
                        enemy_dist: float, attack_range: float,
                        super_range: float, brawler_info: dict, now: float) -> None:

        panic_thresh = self.brain.data.panic_threshold if self.brain else 0.45
        is_pressed = enemy_dist < attack_range * panic_thresh
        is_regenerating = time.time() - self._last_hit_taken < 3.0

        in_range = enemy_dist <= attack_range
        has_los  = self._can_pierce_walls(self.brain.brawler if self.brain else "", "attack") \
                   or self._has_los(player_pos, enemy_pos)
                   
        ammo_ok = True
        if self.brain:
            ammo_ok = self.brain.ammo > 0
            
        cooldown_ok = now - self._last_attack_time > self._attack_cooldown * 0.7

        if in_range and has_los and ammo_ok and (not is_regenerating or is_pressed) and cooldown_ok:
            if self.should_use_gadget and self.is_gadget_ready:
                self.use_gadget()
                self.is_gadget_ready = False

            if self.is_hypercharge_ready:
                self.use_hypercharge()
                self.is_hypercharge_ready = False

            pred = self._predicted_pos(enemy_pos, player_pos, attack_range, now)
            if self.directed_attack(player_pos, pred, attack_range, is_super=False, brain=self.brain):
                self._last_attack_time = now
                if self.brain:
                    self.brain.reward_safe_attack()

        if not self.is_super_ready:
            return
        s_los = self._can_pierce_walls(self.brain.brawler if self.brain else "", "super") \
                or self._has_los(player_pos, enemy_pos)
        super_type = brawler_info.get("super_type","")
        if s_los and (enemy_dist <= super_range or super_type in ("spawnable","other")):
            pred = self._predicted_pos(enemy_pos, player_pos, attack_range, now)
            if self.directed_attack(player_pos, pred, attack_range, is_super=True):
                self.is_super_ready = False

    def _get_movement(self, data: dict, brawler: str) -> str:
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Unknown brawler: {brawler}")

        safe_range, attack_range, super_range = self._get_range(brawler)
        player_pos = self.box_center(data["player"][0])
        now        = time.time()

        enemies = data.get("enemy") or []
        if not enemies:
            self._tracker.reset()
            preferred = random.choice(["w","a","s","d"])
            return self._best_free_move(player_pos, preferred)

        target = self._find_best_target(enemies, player_pos, "attack")
        if target is None:
            self._tracker.reset()
            preferred = random.choice(["w","a","s","d"])
            return self._best_free_move(player_pos, preferred)

        enemy_pos, enemy_dist = target

        if enemy_dist < attack_range * 0.4:
            if self.brain and now - self._last_learn_time > 5:
                self.brain.punish_too_close()
                self._last_learn_time = now

        teammates   = data.get("teammate", [])
        panic_target = self._get_panic_target(player_pos, enemy_pos, enemy_dist, attack_range, teammates)

        is_thrower  = brawler_info.get("ignore_walls_for_attacks", False)
        walls_block = not self._has_los(player_pos, enemy_pos)
        kiting_mul  = self.brain.data.kiting_multiplier if self.brain else 0.85
        ideal_dist  = attack_range * kiting_mul

        if walls_block and not is_thrower and not panic_target:
            dx = enemy_pos[0] - player_pos[0]
            dy = enemy_pos[1] - player_pos[1]
            strafe_time = self.brain.data.strafe_interval if self.brain else 1.5
            sf = 1 if int(now / strafe_time) % 2 == 0 else -1
            vx = dx * 0.5 + dy * sf * 1.5
            vy = dy * 0.5 - dx * sf * 1.5
            movement = self._vector_to_keys(vx, vy, threshold=1.5)
            
        elif is_thrower and walls_block and enemy_dist <= attack_range and not panic_target:
            movement = random.choice(["", "a", "d", "w", "s"]) if int(now/1.5)%2==0 else ""
        else:
            vx, vy   = self._calc_kite_vector(player_pos, enemy_pos, enemy_dist, ideal_dist, panic_target)
            movement = self._vector_to_keys(vx, vy, threshold=1.5)

        if movement and not self._is_path_clear(player_pos, movement):
            alts = [m for m in ["w","a","s","d", "wa", "wd", "sa", "sd"] if m not in movement]
            random.shuffle(alts)
            movement = next((m for m in alts if self._is_path_clear(player_pos, m)), movement)

        self._handle_attack(player_pos, enemy_pos, enemy_dist, attack_range, super_range, brawler_info, now)

        return movement

    def _frame_hash(self, frame):
        return int(np.mean(frame)) if frame is not None else 0

    def _update_walls(self, frame, now: float) -> None:
        tile_data = self.Detect_tiles.detect_objects(frame, conf_tresh=self.wall_det_conf)
        walls: list[Box] = []
        for cls, boxes in tile_data.items():
            if cls != "bush":
                walls.extend(boxes)

        self._wall_history.append(walls)
        if len(self._wall_history) > self._wall_hist_len:
            self._wall_history.pop(0)

        merged: dict[tuple,int] = {}
        for wlist in self._wall_history:
            for w in wlist:
                k = tuple(w)
                merged[k] = merged.get(k, 0) + 1
        combined = [list(k) for k,c in merged.items() if c >= 1]
        self._last_walls = combined
        self._wall_index.rebuild(combined)

    def _check_abilities(self, frame, now: float) -> None:
        w = self.window_controller
        if now - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            crop = frame.crop((
                int(1350*w.width_ratio), int(940*w.height_ratio),
                int(1450*w.width_ratio), int(1050*w.height_ratio)
            ))
            self.is_hypercharge_ready = count_hsv_pixels(crop,(137,158,159),(179,255,255)) > self.hyper_px_min
            self.time_since_hypercharge_checked = now

        if now - self.time_since_gadget_checked > self.gadget_treshold:
            crop = frame.crop((
                int(1580*w.width_ratio), int(930*w.height_ratio),
                int(1700*w.width_ratio), int(1050*w.height_ratio)
            ))
            self.is_gadget_ready = count_hsv_pixels(crop,(57,219,165),(62,255,255)) > self.gadget_px_min
            self.time_since_gadget_checked = now

        if now - self.time_since_super_checked > self.super_treshold:
            crop = frame.crop((
                int(1460*w.width_ratio), int(830*w.height_ratio),
                int(1560*w.width_ratio), int(930*w.height_ratio)
            ))
            self.is_super_ready = count_hsv_pixels(crop,(17,170,200),(27,255,255)) > self.super_px_min
            self.time_since_super_checked = now

    @staticmethod
    def _validate(data: dict) -> Optional[dict]:
        if "player" not in data:
            return None
        data.setdefault("enemy",    None)
        data.setdefault("teammate", [])
        data.setdefault("wall",     [])
        return data

    def main(self, frame, brawler: str) -> None:
        now = time.time()

        if now - self._last_loop_time < self.frame_time:
            return
        self._last_loop_time = now

        with self._frame_lock:
            self._current_frame = frame

        # --- DETECT FRAME FREEZE ---
        current_hash = self._frame_hash(frame)
        if hasattr(self, "_last_frame_hash"):
            if current_hash == self._last_frame_hash:
                if not hasattr(self, "_freeze_start"):
                    self._freeze_start = now
            else:
                self._freeze_start = now
        else:
            self._freeze_start = now

        self._last_frame_hash = current_hash
        is_frozen = now - self._freeze_start > 1.0

        if self.brain is None and brawler:
            self.brain = AdaptiveBrain(brawler)
        if self.brain:
            self.brain.tick_ammo()

        if (self.should_detect_walls and
            now - self._last_wall_proc > self._wall_proc_ivl):
            self._update_walls(frame, now)
            self._last_wall_proc = now

        self._check_abilities(frame, now)

        with self._data_lock:
            data = dict(self._cached_data) if self._cached_data is not None else {}

        if data.get("player"):
            self._last_det["player"] = now
        if data.get("enemy"):
            self._last_det["enemy"] = now

        # --- ANTI-FREEZE & BLIND MOVEMENT ---
        if is_frozen or not data or not data.get("player"):
            
            if now - getattr(self, '_last_state_check_time', 0) > 1.5:
                self._cached_state = get_state(frame)
                self._last_state_check_time = now
            
            if getattr(self, '_cached_state', '') == "match":
                if is_frozen:
                    logger.debug("⚠️ FRAME FROZEN - ВМИКАЮ СЛІПУ АТАКУ!")
                
                if now - getattr(self, '_last_blind_move_time', 0) > 1.5:
                    if self.game_mode == 3:
                        self._blind_move_dir = random.choice(["w", "w", "wa", "wd"])
                    else:
                        self._blind_move_dir = random.choice(["d", "d", "wd", "sd"])
                    self._last_blind_move_time = now
                    
                forced_move = getattr(self, '_blind_move_dir', "w" if self.game_mode == 3 else "d")
                
                # Додаємо трохи хаосу, щоб не впертись у стіну намертво
                if random.random() < 0.2:
                    forced_move = random.choice(["w", "a", "s", "d", "wa", "wd", "sa", "sd"])

                self.do_movement(forced_move)
                
                if now - getattr(self, '_last_attack_time', 0) > 1.0:
                    self.attack_in_direction(forced_move)
                    self._last_attack_time = now
                    
                self._last_no_det_proc = now 
                return
            else:
                self.release_all()
                self._last_det["player"] = now 

            if now - self._last_no_det_proc > self.no_detect_delay:
                current_real_state = get_state(frame)
                if current_real_state == "match":
                    for x, y, sl in [(960,540,0.02),(960,950,0.05),(1660,980,0.08)]:
                        self.window_controller.click(x, y, sl, False)
                self._last_no_det_proc = now
            return

        self._last_no_det_proc = now
        
        valid_data = self._validate(data)
        if valid_data is None:
            return

        movement = self._get_movement(valid_data, brawler)
        movement = self.unstuck_movement_if_needed(movement, now)
        self.do_movement(movement)

        self.scene_data.append({
            "frame": len(self.scene_data),
            "player": valid_data.get("player",[]),
            "enemy":  valid_data.get("enemy", []),
            "wall":   valid_data.get("wall",  []),
            "move":   movement,
        })
        if len(self.scene_data) > 600:
            self.scene_data.pop(0)

    def generate_visualization(self, output: str = "visualization.mp4") -> None:
        W, H = GAME_WIDTH, GAME_HEIGHT
        out  = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W,H))
        font = cv2.FONT_HERSHEY_SIMPLEX

        for fd in self.scene_data:
            img = np.zeros((H, W, 3), np.uint8)
            sx  = W / GAME_WIDTH
            sy  = H / GAME_HEIGHT

            for x1,y1,x2,y2 in fd["wall"]:
                cv2.rectangle(img,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(90,90,90),-1)

            for box in (fd["enemy"] or []):
                x1,y1,x2,y2 = box
                cv2.rectangle(img,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(0,0,220),-1)

            for box in (fd["player"] or []):
                x1,y1,x2,y2 = box
                cv2.rectangle(img,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(0,200,60),-1)

            direction = self._move_to_dir(fd["move"])
            cv2.putText(img, f"move: {direction}", (12, H-12), font, 0.55, (255,255,255), 1)
            out.write(img)

        out.release()
        logger.info("Visualization saved to %s", output)

    @staticmethod
    def _move_to_dir(m: str) -> str:
        table = {
            "":   "idle",
            "w":  "up",    "s":  "down",
            "a":  "left",  "d":  "right",
            "wa": "↖",     "wd": "↗",
            "sa": "↙",     "sd": "↘",
        }
        key = "".join(sorted(m.lower()))
        return table.get(key, m)