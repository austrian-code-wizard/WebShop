import gym
import random
import requests
import string
import time
import json
from io import BytesIO

import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from bs4.element import Comment
from PIL import Image, ImageCms
from os.path import join, dirname, abspath
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import ElementNotInteractableException
from web_agent_site.engine.engine import parse_action, END_BUTTON


class WebAgentDreamEnv(gym.Env):
    """Gym environment for HTML mode of WebShop environment"""

    WINDOW_HEIGHT: int = 540
    WINDOW_WIDTH: int = 960
    DEFAULT_SCROLL_AMOUNT: int = 180
    DEFAULT_SCROLL_TIME: int = 150

    def __init__(self, observation_mode='html', **kwargs):
        """
        Constructor for HTML environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        pause (`float`) -- Pause (in seconds) after taking an action. 
            This is mainly for demo purposes.
            Recommended value: 2.0s
        render (`bool`) -- Show browser if set to `True`.
        session ('str') -- Session ID to initialize environment with
        """
        super().__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        # Create a browser driver to simulate the WebShop site
        options = webdriver.ChromeOptions()
        options.add_argument(f"window-size={self.WINDOW_WIDTH},{self.WINDOW_HEIGHT}")
        options.add_argument('--force-device-scale-factor=1')
        if 'render' not in kwargs or not kwargs['render']:
            options.add_argument("headless")
            options.add_argument("disable-gpu")
            options.add_argument("no-sandbox")
        else:
            raise ValueError("Rendering not supported for Dream environment since it will result in invalid window sizes")
        binary_path = join(dirname(abspath(__file__)), 'chromedriver')
        print(f'Using Chrome binary at {binary_path}')
        self.browser = webdriver.Chrome(executable_path=binary_path, options=options)

        # Set flags and values for WebShop session
        self.text_to_clickable = None
        self.assigned_session = kwargs.get('session')
        self.session = None
        self.click_locations = [
            ()
        ]
        self.reset()

    def step(self, action: int):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        reward = 0.0
        done = False
        info = None

        # Map action to executed command on the WebShop environment via the broswer driver
        clickables = self.get_available_click_actions()
        if action == 0:
            self.execute_search_query_input()
        elif action == 1:
            self.execute_scroll_down()
        elif action == 2:
            self.execute_scroll_up()
        elif action == 3:
            done = True
        elif action - 4 < len(clickables):
            try:
                clickables[action - 4].click()
            except ElementNotInteractableException:
                # Perform force click with JavaScript
                button = clickables[action - 4]
                self.browser.execute_script("arguments[0].click();", button)
        else:
            print('Invalid action. No action performed.')

        if 'pause' in self.kwargs:
            time.sleep(self.kwargs['pause'])
        return self.state, reward, done, info
    
    def get_available_click_actions(self):
        """Returns list of available actions at the current step"""

        # Collect buttons, links, and options as clickables
        buttons = self.browser.find_elements_by_class_name('btn')
        product_links = self.browser.find_elements_by_class_name('product-link')
        # buying_options = self.browser.find_elements_by_css_selector("input[type='radio']")

        clickables = buttons + product_links # + buying_options
        return [c for c in clickables if self.is_element_in_viewport(c)]


    def is_element_in_viewport(self, element):
        return self.browser.execute_script("""
            var rect = arguments[0].getBoundingClientRect();
            return (
                rect.top >= 0 &&
                rect.left >= 0 &&
                rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                rect.right <= (window.innerWidth || document.documentElement.clientWidth)
            );
        """, element)
    

    def _execute_scroll(self, scroll_up: bool = False):
        """Use the scroll wheel to scroll at coordinates (left, top)."""
        chain = ActionChains(self.browser)
        chain.w3c_actions.wheel_action.scroll(
            x=int(self.WINDOW_WIDTH // 2),
            y=int(self.WINDOW_HEIGHT // 2),
            delta_y=-self.DEFAULT_SCROLL_AMOUNT if scroll_up else self.DEFAULT_SCROLL_AMOUNT,
            duration=self.DEFAULT_SCROLL_TIME,
        )
        chain.w3c_actions.perform()


    def execute_scroll_down(self):
        """Scroll down the page"""
        self._execute_scroll(scroll_up=False)


    def execute_scroll_up(self):
        """Scroll up the page"""
        self._execute_scroll(scroll_up=True)


    def execute_search_query_input(self):
        try:
            search_bar = self.browser.find_element_by_id('search_input')
            search_bar.send_keys(self.instruction_text)
        except Exception:
            pass
        

    def _parse_html(self, html=None, url=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            if url is not None:
                html = requests.get(url)
            else:
                html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    def get_reward(self):
        """Get reward value at current step of the environment"""
        html_obj = self._parse_html()
        r = html_obj.find(id='reward')
        r = float(r.findChildren("pre")[0].string) if r is not None else 0.0
        return r
    
    def get_instruction_text(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text
    
    def get_best_products(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.browser.page_source)
        best_products_element = html_obj.find(id='best-products')
        best_products = json.loads(best_products_element.h4.text)
        return best_products
    
    def convert_html_to_text(self, html):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        return visible_texts
    

    @property
    def browser_metadata(self):
        return dict(
            url=self.browser.current_url,
            y_offset=self.browser.execute_script("return window.pageYOffset;"),
            x_offset=self.browser.execute_script("return window.pageXOffset;"),
            height=self.browser.execute_script("return window.innerHeight;"),
            width=self.browser.execute_script("return window.innerWidth;")
        )


    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            html=self.browser.page_source,
            text = self.convert_html_to_text(self.browser.page_source),
            instruction_text=self.instruction_text,
            screenshot=self.screenshot,
            metadata=self.browser_metadata,
            best_products=self.best_products,
            click_actions = [a.text for a in self.get_available_click_actions()]
        )

    @property
    def action_space(self):
        # Recommended to use `get_available_actions` instead
        return NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def reset(self):
        """Create a new session and reset environment variables"""
        if self.assigned_session is not None:
            self.session = self.assigned_session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        init_url = f'http://127.0.0.1:3000/{self.session}'
        self.browser.get(init_url)

        self.instruction_text = self.get_instruction_text()
        self.best_products = self.get_best_products()

        return self.state, None

    def render(self, mode='human'):
        # TODO: Render observation in terminal or WebShop website
        return self.screenshot

    def close(self):
        # TODO: When DB used instead of JSONs, tear down DB here
        self.browser.close()
        print('Browser closed.')

    @property
    def screenshot(
        self
    ) -> Image.Image:
        """Return a scaled and cropped screenshot taken by the Selenium instance.

        Args:
            driver: Chrome WebDriver.
            true_width: Width of the screenshot in the correct resolution.
            true_height: Height of the screenshot in the correct resolution.
            crop_width: Width to crop the image to.
            crop_height: Height to crop the image to.

        Returns:
            A PIL Image object with width crop_width and height crop_height.
        """
        png_data = self.browser.get_screenshot_as_png()
        pil_image = Image.open(BytesIO(png_data)).convert("RGB")
        icc_profile = pil_image.info.get("icc_profile")
        if icc_profile:
            orig_icc = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
            srgb_icc = ImageCms.createProfile("sRGB")
            pil_image = ImageCms.profileToProfile(pil_image, orig_icc, srgb_icc)
        return pil_image

def tag_visible(element):
    """Helper method to strip HTML block of extraneous tags"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and str(element.parent.parent.get("style")) != "display: none;" and not isinstance(element, Comment)
    )
