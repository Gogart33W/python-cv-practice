import os.path
import sys
import asyncio
import time
import cv2
import numpy as np
import pyautogui
import requests

from state_finder.main import get_state
from trophy_observer import TrophyObserver
from utils import find_template_center, extract_text_and_positions, load_toml_as_dict, async_notify_user, \
    save_brawler_data

user_id = load_toml_as_dict("cfg/general_config.toml")['discord_id']
debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
user_webhook = load_toml_as_dict("cfg/general_config.toml")['personal_webhook']


def notify_user(message_type):
    message_data = {
        'content': f"<@{user_id}> Pyla Bot has completed all it's targets !"
    }
    response = requests.post(user_webhook, json=message_data)
    if response.status_code != 204:
        print(f'Failed to send message. Status code: {response.status_code}')


def load_image(image_path, scale_factor):
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

class StageManager:

    def __init__(self, brawlers_data, lobby_automator, window_controller):
        self.states = {
            'shop': self.quit_shop,
            'brawler_selection': self.quit_shop,
            'popup': self.close_pop_up,
            'match': lambda: 0,
            'end': self.end_game,
            'lobby': self.start_game,
            'play_store': self.click_brawl_stars,
            'star_drop': self.click_star_drop
        }
        self.Lobby_automation = lobby_automator
        self.lobby_config = load_toml_as_dict("./cfg/lobby_config.toml")
        self.brawl_stars_icon = None
        self.close_popup_icon = None
        self.brawlers_pick_data = brawlers_data
        brawler_list = [brawler["brawler"] for brawler in brawlers_data]
        self.Trophy_observer = TrophyObserver(brawler_list)
        self.time_since_last_stat_change = time.time()
        self.long_press_star_drop = load_toml_as_dict("./cfg/general_config.toml")["long_press_star_drop"]
        self.window_controller = window_controller

    def start_brawl_stars(self, frame):
        data = extract_text_and_positions(np.array(frame))
        for key in list(data.keys()):
            if key.replace(" ", "") in ["brawl", "brawlstars", "stars"]:
                x, y = data[key]['center']
                self.window_controller.click(x, y)
                return

        brawl_stars_icon_coords = self.lobby_config['lobby'].get('brawl_stars_icon', [960, 540])
        x, y = brawl_stars_icon_coords[0]*self.window_controller.width_ratio, brawl_stars_icon_coords[1]*self.window_controller.height_ratio
        self.window_controller.click(x, y)

    @staticmethod
    def validate_trophies(trophies_string):
        trophies_string = trophies_string.lower()
        while "s" in trophies_string:
            trophies_string = trophies_string.replace("s", "5")
        numbers = ''.join(filter(str.isdigit, trophies_string))

        if not numbers:
            return False
        return int(numbers)

    def start_game(self, data):
        print("state is lobby, starting game")
        values = {
            "trophies": self.Trophy_observer.current_trophies,
            "wins": self.Trophy_observer.current_wins
        }

        type_of_push = self.brawlers_pick_data[0]['type']
        if type_of_push not in values:
            type_of_push = "trophies"
        value = values[type_of_push]
        if value == "" and type_of_push == "wins":
            value = 0
        
        push_current_brawler_till = self.brawlers_pick_data[0]['push_until']
        if push_current_brawler_till == "" and type_of_push == "wins":
            push_current_brawler_till = 300
        if push_current_brawler_till == "" and type_of_push == "trophies":
            push_current_brawler_till = 1000

        if value >= push_current_brawler_till:
            if len(self.brawlers_pick_data) <= 1:
                print("Brawler reached required trophies/wins. No more brawlers. Stopping.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    screenshot = self.window_controller.screenshot()
                    loop.run_until_complete(async_notify_user("bot_is_stuck", screenshot))
                finally:
                    loop.close()
                self.window_controller.keys_up(list("wasd"))
                self.window_controller.close()
                sys.exit(0)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                screenshot = self.window_controller.screenshot()
                loop.run_until_complete(async_notify_user(self.brawlers_pick_data[0]["brawler"], screenshot))
            finally:
                loop.close()
                
            self.brawlers_pick_data.pop(0)
            self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
            self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
            self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
            next_brawler_name = self.brawlers_pick_data[0]['brawler']
            
            if self.brawlers_pick_data[0]["automatically_pick"]:
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                max_attempts = 30
                attempts = 0
                while current_state != "lobby" and attempts < max_attempts:
                    [time.sleep(0.5) for x, y in [[960, 540], [960, 950], [1660, 980]] if not self.window_controller.click(x, y, 0.02, False)]
                    time.sleep(1)
                    screenshot = self.window_controller.screenshot()
                    current_state = get_state(screenshot)
                    attempts += 1
                if attempts < max_attempts:
                    self.Lobby_automation.select_brawler(next_brawler_name)
            else:
                print("Manual mode, waiting 10 seconds to let user switch.")

        self.window_controller.keys_up(list("wasd"))
        self.window_controller.press_key("Q")
        print("Pressed Q to start a match")

    def click_brawl_stars(self, frame):
        screenshot = frame.crop((50, 4, 900, 31))
        if self.brawl_stars_icon is None:
            self.brawl_stars_icon = load_image("state_finder/images_to_detect/brawl_stars_icon.png",
                                               self.window_controller.scale_factor)
        detection = find_template_center(screenshot, self.brawl_stars_icon)
        if detection:
            x, y = detection
            self.window_controller.click(x=x + 50, y=y)
            
    def click_star_drop(self):
        if self.long_press_star_drop == "yes":
            self.window_controller.press_key("Q",10)
        else:
            [time.sleep(0.5) for x, y in [[960, 540], [960, 950], [1660, 980]] if not self.window_controller.click(x, y, 0.02, False)]

    def end_game(self):
        print("\n[SYSTEM] Бій завершено. Обробка результатів Тріо ШД...")
        time.sleep(5) # Даємо час анімації показати кубки та місце
        screenshot = self.window_controller.screenshot()
        
        # --- 1. СИСТЕМА OCR ДЛЯ ТРІО ШОУДАУНУ ---
        data = extract_text_and_positions(np.array(screenshot))
        rank = 4 # Дефолтне місце (якщо нічого не знайшли)
        trophy_delta = 0
        
        for text_key in data.keys():
            t_lower = text_key.lower().replace(" ", "")
            
            # Шукаємо ранг (місце)
            if "rank" in t_lower or "місце" in t_lower or "#" in t_lower:
                nums = ''.join(filter(str.isdigit, t_lower))
                if nums and int(nums) in [1, 2, 3, 4]:
                    rank = int(nums)
            
            # Шукаємо зміну трофеїв (+X або -X)
            if "+" in t_lower or "-" in t_lower:
                nums = ''.join(filter(str.isdigit, t_lower))
                if nums:
                    if "+" in t_lower:
                        trophy_delta = int(nums)
                    elif "-" in t_lower:
                        trophy_delta = -int(nums)

        print(f"[TRIO SD] Зчитано з екрану -> Місце: {rank} | Базові кубки: {trophy_delta}")

        # --- 2. ЛОГІКА ВІНСТРІКУ ---
        brawler = self.brawlers_pick_data[0]
        
        if "win_streak" not in brawler or brawler["win_streak"] == "":
            brawler["win_streak"] = 0
        else:
            brawler["win_streak"] = int(brawler["win_streak"])

        ws_bonus = 0
        
        if rank in [1, 2]:
            brawler["win_streak"] += 1
            print(f"✅ Перемога! Вінстрік збільшено до: {brawler['win_streak']}")
            # Обчислюємо бонус вінстріку
            ws = brawler["win_streak"]
            if ws >= 5: ws_bonus = 4
            elif ws == 4: ws_bonus = 3
            elif ws == 3: ws_bonus = 2
            elif ws == 2: ws_bonus = 1
        elif rank == 3:
            print(f"🤝 Нічия! Вінстрік збережено: {brawler['win_streak']}")
            # Для нічиєї бонус не дається
        elif rank == 4:
            brawler["win_streak"] = 0
            print("❌ Поразка! Вінстрік скинуто до 0.")

        # --- 3. ЗАРАХУВАННЯ ТРОФЕЇВ ---
        total_gained = trophy_delta + ws_bonus
        
        if "trophies" in brawler and isinstance(brawler["trophies"], int):
            brawler["trophies"] += total_gained
            self.Trophy_observer.current_trophies = brawler["trophies"]
            
        print(f"[ТРОФЕЇ] База: {trophy_delta} | Вінстрік: +{ws_bonus} | Разом: {total_gained}")
        print(f"[ТРОФЕЇ] Поточний баланс кубків: {brawler['trophies']}\n")

        save_brawler_data(self.brawlers_pick_data)
        self.time_since_last_stat_change = time.time()

        # --- 4. НАВЧАННЯ НЕЙРОНКИ ---
        if hasattr(self, 'Play') and hasattr(self.Play, 'on_game_result'):
            self.Play.on_game_result(rank)

        # --- 5. ПЕРЕВІРКА ЦІЛІ (Push Until) ---
        type_to_push = brawler.get('type', 'trophies')
        target_val = brawler.get('push_until', 1000)
        if target_val == "": target_val = 1000
        current_val = brawler.get(type_to_push, 0)
        
        if current_val >= target_val:
            if len(self.brawlers_pick_data) <= 1:
                print("🏆 Ціль досягнута! Немає більше бравлерів у черзі. Зупиняю бота.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    screenshot = self.window_controller.screenshot()
                    loop.run_until_complete(async_notify_user("completed", screenshot))
                finally:
                    loop.close()
                if os.path.exists("latest_brawler_data.json"):
                    os.remove("latest_brawler_data.json")
                self.window_controller.keys_up(list("wasd"))
                self.window_controller.close()
                sys.exit(0)

        # --- 6. ВИХІД У ЛОБІ ---
        current_state = get_state(screenshot)
        max_end_attempts = 30
        end_attempts = 0
        while current_state == "end" and end_attempts < max_end_attempts:
            # Клікаємо кнопку Продовжити (Exit)
            [time.sleep(0.5) for x, y in [[960, 540], [960, 950], [1660, 980]] if not self.window_controller.click(x, y, 0.02, False)]
            time.sleep(3)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            end_attempts += 1
            
        if end_attempts >= max_end_attempts:
            print("Екран завершення завис, примусовий вихід...")
        if debug: print("Game has ended", current_state)

    def quit_shop(self):
        self.window_controller.click(100*self.window_controller.width_ratio, 60*self.window_controller.height_ratio)

    def close_pop_up(self):
        screenshot = self.window_controller.screenshot()
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image("state_finder/images_to_detect/close_popup.png", self.window_controller.scale_factor)
        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)

    def do_state(self, state, data=None):
        if data is not None:
            self.states[state](data)
            return
        self.states[state]()