import math
import random
import time

import cv2
import numpy as np
from PIL import Image
from state_finder.main import get_state
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info

brawl_stars_width, brawl_stars_height = 1920, 1080

# Запобіжники для конфігів
gen_cfg = load_toml_as_dict("cfg/general_config.toml")
debug = gen_cfg.get('super_debug', 'no') == "yes"

lobby_cfg = load_toml_as_dict("./cfg/lobby_config.toml")
pixel_areas = lobby_cfg.get('pixel_counter_crop_area', {})
super_crop_area = pixel_areas.get('super', [1460, 830, 1560, 930])
gadget_crop_area = pixel_areas.get('gadget', [1580, 930, 1700, 1050])
hypercharge_crop_area = pixel_areas.get('hypercharge', [1350, 940, 1450, 1050])


class Movement:

    def __init__(self, window_controller):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config.get("unstuck_movement_delay", 3.0),
            "duration": bot_config.get("unstuck_movement_hold_time", 1.5),
            "toggled": False,
            "started_at": time.time(),
            "fixed": ""
        }
        self.game_mode = bot_config.get("gamemode_type", 3)
        self.gamemode_str = str(bot_config.get("gamemode", "other")).lower()
        self.is_showdown = "sd" in self.gamemode_str or "showdown" in self.gamemode_str
        
        gadget_value = bot_config.get("bot_uses_gadgets", "yes")
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.super_treshold = time_config.get("super", 5.0)
        self.gadget_treshold = time_config.get("gadget", 5.0)
        self.hypercharge_treshold = time_config.get("hypercharge", 5.0)
        self.walls_treshold = time_config.get("wall_detection", 1.0)
        self.keep_walls_in_memory = self.walls_treshold <= 1
        self.last_walls_data = []
        self.keys_hold = []
        self.time_since_different_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.window_controller = window_controller
        self.TILE_SIZE = 60
        
    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    @staticmethod
    def get_horizontal_move_key(direction_x, opposite=False):
        if opposite:
            return "A" if direction_x > 0 else "D"
        return "D" if direction_x > 0 else "A"

    @staticmethod
    def get_vertical_move_key(direction_y, opposite=False):
        if opposite:
            return "W" if direction_y > 0 else "S"
        return "S" if direction_y > 0 else "W"

    def attack(self, touch_up=True, touch_down=True):
        self.window_controller.press_key("M", touch_up=touch_up, touch_down=touch_down)

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press_key("H")

    def use_gadget(self):
        print("Using gadget")
        self.window_controller.press_key("G")

    def use_super(self):
        print("Using super")
        self.window_controller.press_key("E")

    @staticmethod
    def get_random_attack_key():
        random_movement = random.choice(["A", "W", "S", "D"])
        random_movement += random.choice(["A", "W", "S", "D"])
        return random_movement

    @staticmethod
    def reverse_movement(movement):
        movement = movement.lower()
        translation_table = str.maketrans("wasd", "sdwa")
        return movement.translate(translation_table)

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()
        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False

            return self.fix_movement_keys['fixed']

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            self.time_since_different_movement = current_time

        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            reversed_movement = self.reverse_movement(movement)

            if reversed_movement == "s":
                reversed_movement = random.choice(['aw', 'dw'])
            elif reversed_movement == "w":
                reversed_movement = random.choice(['as', 'ds'])

            self.fix_movement_keys['fixed'] = reversed_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            return reversed_movement

        return movement


class Play(Movement):

    def __init__(self, main_info_model, tile_detector_model, window_controller):
        super().__init__(window_controller)

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main_info = Detect(main_info_model, classes=['enemy', 'teammate', 'player', 'box'])
        self.tile_detector_model_classes = bot_config.get("wall_model_classes", [])
        self.Detect_tile_detector = Detect(
            tile_detector_model,
            classes=self.tile_detector_model_classes
        )

        self.time_since_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.time_since_hypercharge_checked = time.time()
        self.time_since_super_checked = time.time()
        self.time_since_walls_checked = 0
        self.time_since_movement_change = time.time()
        self.time_since_player_last_found = time.time()
        self.current_brawler = None
        self.is_hypercharge_ready = False
        self.is_gadget_ready = False
        self.is_super_ready = False
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement = ''
        self.last_movement_time = time.time()
        self.wall_history = []
        self.wall_history_length = 3  
        self.scene_data = []
        self.should_detect_walls = bot_config.get("gamemode", "") in ["brawlball", "brawl_ball", "brawll ball"]
        self.minimum_movement_delay = bot_config.get("minimum_movement_delay", 0.1)
        self.no_detection_proceed_delay = time_config.get("no_detection_proceed", 2.0)
        self.gadget_pixels_minimum = bot_config.get("gadget_pixels_minimum", 2000.0)
        self.hypercharge_pixels_minimum = bot_config.get("hypercharge_pixels_minimum", 2000.0)
        self.super_pixels_minimum = bot_config.get("super_pixels_minimum", 2400.0)
        self.wall_detection_confidence = bot_config.get("wall_detection_confidence", 0.9)
        self.entity_detection_confidence = bot_config.get("entity_detection_confidence", 0.6)
        self.time_since_holding_attack = None
        self.seconds_to_hold_attack_after_reaching_max = bot_config.get("seconds_to_hold_attack_after_reaching_max", 0.2)

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info.get('attack_range', 600)
            safe_range = info.get('safe_range', 400)
            super_range = info.get('super_range', 600)
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info.get(brawler, {}).get('ignore_walls_for_attacks', False)
        elif skill_type == "super":
            return brawlers_info.get(brawler, {}).get('ignore_walls_for_supers', False)
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        return brawlers_info.get(brawler, {}).get('hold_attack', 0) > 0

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls):
        if not walls:
            return False

        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x, max_x = min(p1_t[0], p2_t[0]), max(p1_t[0], p2_t[0])
        min_y, max_y = min(p1_t[1], p2_t[1]), max(p1_t[1], p2_t[1])
        for wall in walls:
            x1, y1, x2, y2 = wall

            if max_x < x1 or min_x > x2 or max_y < y1 or min_y > y2:
                continue

            rect = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                return True
        return False

    def no_enemy_movement(self, player_data, walls):
        player_position = self.get_player_pos(player_data)
        
        # Адаптивний рух для ШД
        if self.is_showdown:
            preferred_movement = random.choice(['W', 'A', 'S', 'D', 'WA', 'WD', 'SA', 'SD'])
        else:
            preferred_movement = 'W' if self.game_mode == 3 else 'D'

        if not self.is_path_blocked(player_position, preferred_movement, walls):
            return preferred_movement
        else:
            alternative_moves = ['W', 'A', 'S', 'D']
            if preferred_movement in alternative_moves:
                alternative_moves.remove(preferred_movement)
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_position, move, walls):
                    return move
            return preferred_movement

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        if self.walls_block_line_of_sight(player_pos, enemy_pos, walls):
            return False
        return True

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            return closest_hittable
        elif closest_unhittable:
            return closest_unhittable

        return None, None

    def get_main_data(self, frame):
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        return data

    def is_path_blocked(self, player_pos, move_direction, walls, distance=None):
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= distance
        if 's' in move_direction.lower():
            dy += distance
        if 'a' in move_direction.lower():
            dx -= distance
        if 'd' in move_direction.lower():
            dx += distance
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        return self.walls_block_line_of_sight(player_pos, new_pos, walls)

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if "player" not in data.keys():
            incomplete = True

        if "enemy" not in data.keys():
            data['enemy'] = None

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        if 'box' not in data.keys():
            data['box'] = None

        return False if incomplete else data

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    # 🔥 ОРИГІНАЛЬНИЙ РУХ 🔥 (Затискає кнопки кожний кадр, як і треба емулятору)
    def do_movement(self, movement):
        movement = movement.lower()
        keys_to_keyDown = []
        keys_to_keyUp = []
        for key in ['w', 'a', 's', 'd']:
            if key in movement:
                keys_to_keyDown.append(key)
            else:
                keys_to_keyUp.append(key)

        if keys_to_keyDown:
            self.window_controller.keys_down(keys_to_keyDown)

        self.window_controller.keys_up(keys_to_keyUp)
        self.keys_hold = keys_to_keyDown

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    # ДЕТЕКТОР ОТРУЙНОГО ДИМУ
    def get_gas_vector(self, frame, player_pos):
        if frame is None or not self.is_showdown: return 0.0, 0.0
        try:
            arr = np.array(frame) if isinstance(frame, Image.Image) else frame
            small = cv2.resize(arr, (192, 108))
            hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([45, 150, 150]), np.array([85, 255, 255]))
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) < 10: return 0.0, 0.0 
                
            px, py = player_pos[0] / 10.0, player_pos[1] / 10.0
            gx, gy = 0.0, 0.0
            fear_radius = 45.0 * self.window_controller.scale_factor
            
            for x, y in zip(x_coords, y_coords):
                if math.hypot(x - px, y - py) < fear_radius:
                    weight = 1.0 / max(math.hypot(x - px, y - py), 1.0)
                    gx += (px - x) * weight; gy += (py - y) * weight
                    
            if math.hypot(gx, gy) > 0:
                return (gx / math.hypot(gx, gy)) * 500.0, (gy / math.hypot(gx, gy)) * 500.0
        except Exception: pass
        return 0.0, 0.0

    def get_movement(self, player_data, enemy_data, walls, brawler, frame=None):
        brawler_info = self.brawlers_info.get(brawler, {})
        must_brawler_hold_attack = self.must_brawler_hold_attack(brawler, self.brawlers_info)
        
        if must_brawler_hold_attack and self.time_since_holding_attack is not None and time.time() - self.time_since_holding_attack >= brawler_info.get('hold_attack', 0) + self.seconds_to_hold_attack_after_reaching_max:
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)
        
        # 1. ТІКАЄМО ВІД ДИМУ
        gas_vx, gas_vy = self.get_gas_vector(frame, player_pos)
        if math.hypot(gas_vx, gas_vy) > 0:
            return ("D" if gas_vx > 0 else "A") + ("S" if gas_vy > 0 else "W")

        # 2. РУХ
        if not self.is_there_enemy(enemy_data):
            return self.no_enemy_movement(player_data, walls)
            
        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        
        if enemy_coords is None:
            return self.no_enemy_movement(player_data, walls)

        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        if enemy_distance > safe_range:
            move_horizontal = self.get_horizontal_move_key(direction_x)
            move_vertical = self.get_vertical_move_key(direction_y)
        else:
            move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
            move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

        movement_options = [move_horizontal + move_vertical]
        if self.game_mode == 3:
            movement_options += [move_vertical, move_horizontal]
        elif self.game_mode == 5 or self.is_showdown:
            movement_options += [move_horizontal, move_vertical]
        else:
            movement_options += [move_vertical, move_horizontal]

        for move in movement_options:
            if not self.is_path_blocked(player_pos, move, walls):
                movement = move
                break
        else:
            alternative_moves = ['W', 'A', 'S', 'D']
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break
            else:
                movement = move_horizontal + move_vertical

        current_time = time.time()
        if movement != self.last_movement:
            if current_time - self.last_movement_time >= self.minimum_movement_delay:
                self.last_movement = movement
                self.last_movement_time = current_time
            else:
                movement = self.last_movement 
        else:
            self.last_movement_time = current_time

        # 3. АТАКИ ТА УЛЬТИ
        if self.is_super_ready and self.time_since_holding_attack is None:
            super_type = brawler_info.get('super_type', '')
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")

            if (enemy_hittable and
                    (enemy_distance <= super_range
                     or super_type in ["spawnable", "other"]
                     or (brawler in ["stu", "surge"] and super_type == "charge" and enemy_distance <= super_range + attack_range)
                    )):
                if self.is_hypercharge_ready:
                    self.use_hypercharge()
                    self.time_since_hypercharge_checked = time.time()
                    self.is_hypercharge_ready = False
                self.use_super()
                self.time_since_super_checked = time.time()
                self.is_super_ready = False

        if enemy_distance <= attack_range:
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
            if enemy_hittable:
                if self.should_use_gadget == True and self.is_gadget_ready and self.time_since_holding_attack is None:
                    self.use_gadget()
                    self.time_since_gadget_checked = time.time()
                    self.is_gadget_ready = False

                if not must_brawler_hold_attack:
                    self.attack()
                else:
                    if self.time_since_holding_attack is None:
                        self.time_since_holding_attack = time.time()
                        self.attack(touch_up=False, touch_down=True)
                    elif time.time() - self.time_since_holding_attack >= brawler_info.get('hold_attack', 0):
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None

        return movement

    def loop(self, brawler, data, current_time, frame=None):
        # АТАКА НА КОРОБКИ В ШД
        if self.is_showdown and data.get("box") and not data.get("enemy"):
            player_pos = self.get_player_pos(data['player'][0])
            box_target = min(data['box'], key=lambda b: math.hypot(self.get_enemy_pos(b)[0]-player_pos[0], self.get_enemy_pos(b)[1]-player_pos[1]))
            box_pos = self.get_enemy_pos(box_target)
            box_dist = math.hypot(box_pos[0]-player_pos[0], box_pos[1]-player_pos[1])
            
            _, attack_range, _ = self.get_brawler_range(brawler)
            
            if box_dist < attack_range:
                self.attack()
                
            direction_x = box_pos[0] - player_pos[0]
            direction_y = box_pos[1] - player_pos[1]
            move_horizontal = self.get_horizontal_move_key(direction_x)
            move_vertical = self.get_vertical_move_key(direction_y)
            movement = move_horizontal + move_vertical
        else:
            movement = self.get_movement(player_data=data['player'][0], enemy_data=data['enemy'], walls=data['wall'], brawler=brawler, frame=frame)
            
        current_time = time.time()
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            movement = self.unstuck_movement_if_needed(movement, current_time)
            self.do_movement(movement)
            self.time_since_movement = time.time()
        return movement

    def check_if_hypercharge_ready(self, frame):
        frame_np = np.array(frame) if isinstance(frame, Image.Image) else frame
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(hypercharge_crop_area[0] * wr), int(hypercharge_crop_area[1] * hr)
        x2, y2 = int(hypercharge_crop_area[2] * wr), int(hypercharge_crop_area[3] * hr)
        screenshot = frame_np[y1:y2, x1:x2]
        purple_pixels = count_hsv_pixels(screenshot, (137, 158, 159), (179, 255, 255))
        if purple_pixels > self.hypercharge_pixels_minimum: return True
        return False

    def check_if_gadget_ready(self, frame):
        frame_np = np.array(frame) if isinstance(frame, Image.Image) else frame
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(gadget_crop_area[0] * wr), int(gadget_crop_area[1] * hr)
        x2, y2 = int(gadget_crop_area[2] * wr), int(gadget_crop_area[3] * hr)
        screenshot = frame_np[y1:y2, x1:x2]
        green_pixels = count_hsv_pixels(screenshot, (57, 219, 165), (62, 255, 255))
        if green_pixels > self.gadget_pixels_minimum: return True
        return False

    def check_if_super_ready(self, frame):
        frame_np = np.array(frame) if isinstance(frame, Image.Image) else frame
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(super_crop_area[0] * wr), int(super_crop_area[1] * hr)
        x2, y2 = int(super_crop_area[2] * wr), int(super_crop_area[3] * hr)
        screenshot = frame_np[y1:y2, x1:x2]
        yellow_pixels = count_hsv_pixels(screenshot, (17, 170, 200), (27, 255, 255))
        if yellow_pixels > self.super_pixels_minimum: return True
        return False

    def get_tile_data(self, frame):
        return self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)

    def process_tile_data(self, tile_data):
        walls = []
        for class_name, boxes in tile_data.items():
            if class_name != 'bush':
                walls.extend(boxes)
        self.wall_history.append(walls)
        if len(self.wall_history) > self.wall_history_length:
            self.wall_history.pop(0)
        return self.combine_walls_from_history()

    def combine_walls_from_history(self):
        unique_walls = {tuple(wall) for walls in self.wall_history for wall in walls}
        return list(unique_walls)

    def main(self, frame, brawler):
        self.current_brawler = brawler
        current_time = time.time()
        data = self.get_main_data(frame)
        
        if self.should_detect_walls and current_time - self.time_since_walls_checked > self.walls_treshold:
            tile_data = self.get_tile_data(frame)
            walls = self.process_tile_data(tile_data)
            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            data['wall'] = walls
        elif self.keep_walls_in_memory:
            data['wall'] = self.last_walls_data

        data = self.validate_game_data(data)
        self.track_no_detections(data)
        
        if data:
            self.time_since_player_last_found = time.time()
                    
        if not data:
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.keys_up(list("wasd"))
            self.time_since_different_movement = time.time()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                # АНТИ-АФК (Кікстарт на спавні)
                move_dir = random.choice(["w", "a", "s", "d"]) if self.is_showdown else "w"
                self.do_movement(move_dir)
                self.time_since_last_proceeding = time.time()
            return
            
        self.time_since_last_proceeding = time.time()
        self.is_hypercharge_ready = False
        if current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
            
        self.is_gadget_ready = False
        if current_time - self.time_since_gadget_checked > self.gadget_treshold:
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
            
        self.is_super_ready = False
        if current_time - self.time_since_super_checked > self.super_treshold:
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time

        movement = self.loop(brawler, data, current_time, frame=frame)