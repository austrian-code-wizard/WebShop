import random
import requests
import string
import re
import os
import json
import html
import time
import logging
import multiprocessing
from urllib.parse import unquote, quote
from os.path import join, dirname, abspath
from selenium.webdriver import Chrome, ChromeOptions

import gymnasium as gym
from bs4 import BeautifulSoup
from bs4.element import Comment
from PIL import Image, ImageFont, ImageDraw
from string import ascii_letters, digits, punctuation
from web_agent_site.app import app

SEARCH_STATE = 0
FILLED_SEARCH_STATE = 1
RESULTS_STATE = 2


def start_app(port:int = 3000, debug:bool = False, host:str = "localhost"):
    app.run(port=port, debug=debug, host=host)


class WebAgentDreamDOMEnv(gym.Env):

    """Gym environment for HTML mode of WebShop environment"""

    _browser = None
    _app_process = None

    APP_PORT = random.randint(5000, 8000)
    APP_HOST = "0.0.0.0"
    TIMEOUT = 0.05

    def __init__(
        self,
        observation_mode: str = 'html',
        window_height: int = 540,
        window_width: int = 960,
        scroll_amount: int = 180,
        scroll_time: int = 150,
        return_n: int = 3,
        num_random: int = 0,
        shuffle_products: bool = False,
        **kwargs
    ):
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
        self.WINDOW_HEIGHT = window_height
        self.WINDOW_WIDTH = window_width
        self.return_n = return_n
        self.num_random = num_random
        self.shuffle_products = shuffle_products
        
        self._cur_state = None
        self.page_source = None
        self.first_step = True

        self.observation_space = gym.spaces.Text(
            min_length=0,
            max_length=100000,
            charset=ascii_letters + digits + punctuation
        )

        if WebAgentDreamDOMEnv._app_process is None:
            # We do not want to output api call logs
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            WebAgentDreamDOMEnv._app_process = multiprocessing.Process(target=start_app, kwargs={'port': WebAgentDreamDOMEnv.APP_PORT, "debug": False, "host": WebAgentDreamDOMEnv.APP_HOST})
            WebAgentDreamDOMEnv._app_process.start()
            print(f"Started app subprocess on {self.APP_HOST}:{self.APP_PORT}")
            time.sleep(1)

        if WebAgentDreamDOMEnv._browser is None:
            options = ChromeOptions()
            options.add_argument(f"window-size={self.WINDOW_WIDTH},{self.WINDOW_HEIGHT}")
            options.add_argument('force-device-scale-factor=0.07')
            options.add_argument("headless")
            options.add_argument("disable-gpu")
            options.add_argument("no-sandbox")
            cur_path = dirname(abspath(__file__))
            # Find all files in cur_path prefixed with 'chromedriver'
            binary_paths = [join(cur_path, f) for f in os.listdir(cur_path) if f.startswith('chromedriver')]

            success = False
            for binary_path in binary_paths:
                try:
                    WebAgentDreamDOMEnv._browser = Chrome(executable_path=binary_path, options=options)
                    print("Started browser")
                    time.sleep(1)
                    success = True
                except:
                    pass
            if not success:
                raise Exception("Could not start browser")
            

    def clean_url(self, url: str) -> str:
        """Make a url nicely readable by adding ellipses in long segments"""
        parts = url.split('/')
        parts = [p[:10] + '...' + p[-10:] if len(p) > 20 else p for p in parts]
        return '/'.join(parts)
    
    def add_query_parameters(self, url_string: str) -> str:
        """
        Adds query parameters 'return_n' and 'num_random' to the given URL string

        Arguments:
        url_string (str): The URL string to add query parameters to
        return_n (int): The value of the 'return_n' query parameter
        num_random (int): The value of the 'num_random' query parameter

        Returns:
        str: The modified URL string with the added query parameters
        """
        if '?' in url_string:
            url_string += '&'
        else:
            url_string += '?'
        url_string += f'return_n={self.return_n}&num_random={self.num_random}'
        url_string += f"&seed={self.seed}" if self.seed is not None else ''
        url_string += f"&shuffle_products={int(self.shuffle_products)}" if self.shuffle_products is not None else ''
        return url_string

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
        self.first_step = False

        # Map action to executed command on the WebShop environment via the broswer driver
        urls = self.get_available_click_actions()
        if action >= len(urls):
            pass
        else:
            with app.test_client() as c:
                self.page_source = c.get(urls[action]).data.decode('utf-8')
                self._cur_state = urls[action]
        return self.state, reward, done, info
    
    def get_available_click_actions(self):
        """Returns list of available actions at the current step"""

        pattern = re.compile(r"action=\"(.*?)\"")
        urls = pattern.findall(self.page_source)
        pattern2 = re.compile(r"class=\"product-link\" href=\"(.*?)\"")
        urls += pattern2.findall(self.page_source)
        urls = [url for url in urls if url.startswith('/')]
        urls = [unquote(url) for url in urls]
        urls = [html.unescape(url) for url in urls]
        return [quote(url) for url in urls]


    def is_element_in_viewport(self, element):
        return True


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
        html_obj = self._parse_html(self.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text
    
    def get_best_products(self):
        """Get corresponding instruction text for environment current step"""
        html_obj = self._parse_html(self.page_source)
        best_products_element = html_obj.find(id='best-products')
        best_products = json.loads(best_products_element.h4.text)
        return best_products
    
    def convert_html_to_text(self, html):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        return visible_texts


    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            html=self.page_source,
            # text = self.convert_html_to_text(self.browser.page_source),
            instruction_text=self.instruction_text,
            # screenshot=self.screenshot,
            # metadata=self.browser_metadata,
            best_products=self.best_products,
            # click_actions = [a.text for a in self.get_available_click_actions()]
        )

    def reset(self, goal_id: int = None, seed: int = 0, is_test: bool = False):
        """Create a new session and reset environment variables"""
        if goal_id is not None:
            self.session = f"fixed_{goal_id}"
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        self.seed = seed
        self.is_test = is_test
        self.first_step = True
        
        with app.test_client() as c:
            url = self.add_query_parameters(f'http://localhost/{self.session}')
            self._cur_state = url.replace(f'http://localhost', '')
            url += "&episode_start=1"
            self.page_source = c.get(url).data.decode('utf-8')

        self.instruction_text = self.get_instruction_text()
        self.best_products = self.get_best_products()

        return self.state, None

    def render(self, mode='human'):
        # TODO: Render observation in terminal or WebShop website
        
        # Create a white PIL image and place the self._curstate url text on it, taking account of overflow
        if self.is_test:
            return self.screenshot
        return self.fake_screenshot

    @classmethod
    def close(cls):
        # TODO: When DB used instead of JSONs, tear down DB here
        cls._browser.quit()
        print(f"Quit driver successfully")
        cls._app_process.terminate()
        cls._app_process.kill()
        cls._app_process.join()
        cls._app_process = None
        print(f"Quit flask")

    @property
    def fake_screenshot(self):
        img = Image.new('RGB', (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), color = (255, 255, 255))
        fnt = ImageFont.truetype('arial.ttf', 12)
        d = ImageDraw.Draw(img)
        d.text((10,10), self.clean_url(self._cur_state), font=fnt, fill=(0,0,0))
        return img
    
    @property
    def screenshot(self):
        """Take screenshot of current browser window"""
        url  = f'http://{self.APP_HOST}:{self.APP_PORT}' + self._cur_state
        if self.first_step:
            url += "&episode_start=1"
        WebAgentDreamDOMEnv._browser.get(url)
        time.sleep(WebAgentDreamDOMEnv.TIMEOUT)
        self._browser.save_screenshot(f'.{self.APP_PORT}-tmp.png')
        time.sleep(WebAgentDreamDOMEnv.TIMEOUT)
        return Image.open(f'.{self.APP_PORT}-tmp.png')

def tag_visible(element):
    """Helper method to strip HTML block of extraneous tags"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and str(element.parent.parent.get("style")) != "display: none;" and not isinstance(element, Comment)
    )
