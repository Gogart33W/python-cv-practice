import time
import numpy as np

from stage_manager import load_image
from typization import BrawlerName
from utils import extract_text_and_positions, count_hsv_pixels, load_toml_as_dict, find_template_center

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"

class LobbyAutomation:

    def __init__(self, window_controller):
        self.coords_cfg = load_toml_as_dict("./cfg/lobby_config.toml")
        self.window_controller = window_controller

    def handle_popups(self, screenshot):
        try:
            img = screenshot.resize((int(screenshot.width * 0.65), int(screenshot.height * 0.65)))
            img = np.array(img)

            results = extract_text_and_positions(img)

            for key, value in results.items():
                # 🔥 Очищуємо текст від сміття, щоб точно вловити "letsgo"
                text = key.lower().replace(" ", "").replace("-", "").replace(".", "").replace("'", "").replace("!", "")

                is_popup = False
                
                if text in ["ok", "okay", "next", "claim", "collect", "letsgo", "letsgо", "continue", "reload", "reward", "rewards"]:
                    is_popup = True
                else:
                    for trigger in ["continue", "claim", "collect", "letsgo", "reload", "letsgо", "reward"]:
                        if trigger in text:
                            is_popup = True
                            break

                if is_popup:
                    if debug:
                        print(f"[POPUP DETECTED] -> {text}")

                    x, y = value['center']
                    self.window_controller.click(int(x * 1.5385), int(y * 1.5385))
                    time.sleep(1)
                    return True

        except Exception as e:
            if debug:
                print("Popup handler error:", e)

        return False

    def check_for_idle(self, frame):
        screenshot = frame
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio

        if self.handle_popups(screenshot):
            return

        screenshot = screenshot.crop(
            (int(400 * wr), int(380 * hr), int(1500 * wr), int(700 * hr)))

        gray_pixels = count_hsv_pixels(screenshot, (0, 0, 55), (10, 15, 77))

        if debug:
            print("gray pixels (if > 850 then bot will try to unidle) :", gray_pixels)

        if gray_pixels > 850:
            self.window_controller.click(int(535 * wr), int(615 * hr))

    def select_brawler(self, brawler):
        self.window_controller.screenshot()
        brawler_menu_treshold = 0.8
        found = False

        while not found:
            screenshot = self.window_controller.screenshot()

            if self.handle_popups(screenshot):
                continue

            brawler_menu_btn_coords = find_template_center(
                screenshot,
                load_image(
                    r'state_finder/images_to_detect/brawler_menu_btn.png',
                    self.window_controller.scale_factor
                ),
                brawler_menu_treshold
            )

            if brawler_menu_btn_coords:
                found = True
            else:
                if debug:
                    print("Brawler menu button not found, retrying...")
                brawler_menu_treshold -= 0.1
                time.sleep(1)

            if not found and brawler_menu_treshold < 0.5:
                image = self.window_controller.screenshot()
                image.save(r'brawler_menu_btn_not_found.png')
                raise ValueError("Brawler menu button not found on screen.")

        x, y = brawler_menu_btn_coords
        self.window_controller.click(x, y)

        c = 0
        found_brawler = False

        for i in range(50):
            screenshot = self.window_controller.screenshot()

            if self.handle_popups(screenshot):
                continue

            screenshot = screenshot.resize((int(screenshot.width * 0.65), int(screenshot.height * 0.65)))
            screenshot = np.array(screenshot)

            if debug:
                print("extracting text on current screen...")

            results = extract_text_and_positions(screenshot)
            reworked_results = {}

            for key in results.keys():
                orig_key = key
                for symbol in [' ', '-', '.', "&"]:
                    key = key.replace(symbol, "")

                key = self.resolve_ocr_typos(key)
                reworked_results[key] = results[orig_key]

            if debug:
                print("All detected text:", reworked_results.keys())
                print()

            if brawler in reworked_results.keys():
                if debug:
                    print("Found brawler ", brawler)

                x, y = reworked_results[brawler]['center']
                self.window_controller.click(int(x * 1.5385), int(y * 1.5385))
                time.sleep(1)

                select_x, select_y = self.coords_cfg['lobby']['select_btn'][0], self.coords_cfg['lobby']['select_btn'][1]
                self.window_controller.click(select_x, select_y, already_include_ratio=False)

                time.sleep(0.5)

                if debug:
                    print("Selected brawler ", brawler)

                found_brawler = True
                break

            # ---------------------------------------------------------
            # ЗАХИСТ ВІД МІСКЛІКІВ
            # Якщо випадково тапнули на перса замість скролу - виходимо назад
            # ---------------------------------------------------------
            in_brawler_profile = False
            for profile_word in ['health', 'damage', 'upgrade', 'super', 'attack']:
                if any(profile_word in k for k in reworked_results.keys()):
                    in_brawler_profile = True
                    break
            
            if in_brawler_profile:
                if debug:
                    print("ACCIDENTAL CLICK DETECTED (stuck in Brawler Profile). Pressing Back button...")
                wr = self.window_controller.width_ratio
                hr = self.window_controller.height_ratio
                self.window_controller.click(int(100 * wr), int(100 * hr))
                time.sleep(1.5)
                continue
            # ---------------------------------------------------------

            if c == 0:
                wr = self.window_controller.width_ratio
                hr = self.window_controller.height_ratio
                self.window_controller.swipe(
                    int(1700 * wr), int(900 * hr),
                    int(1700 * wr), int(850 * hr),
                    duration=0.8
                )
                c += 1
                continue

            wr = self.window_controller.width_ratio
            hr = self.window_controller.height_ratio

            self.window_controller.swipe(
                int(1700 * wr), int(900 * hr),
                int(1700 * wr), int(650 * hr),
                duration=0.8
            )
            time.sleep(1)

        if not found_brawler:
            print(f"WARNING: Brawler '{brawler}' not found.")
            raise ValueError(f"Brawler '{brawler}' could not be found.")

    @staticmethod
    def resolve_ocr_typos(potential_brawler_name: str) -> str:
        matched_typo: str | None = {
            'shey': BrawlerName.Shelly.value,
            'shlly': BrawlerName.Shelly.value,
            'larryslawrie': BrawlerName.Larry.value,
        }.get(potential_brawler_name, None)

        return matched_typo or potential_brawler_name