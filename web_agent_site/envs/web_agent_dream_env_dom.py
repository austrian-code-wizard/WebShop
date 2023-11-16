import gym
import random
import requests
import string
import re
import json
import html
from urllib.parse import unquote, quote

from bs4 import BeautifulSoup
from bs4.element import Comment
from PIL import Image, ImageFont, ImageDraw
from web_agent_site.app import app

SEARCH_STATE = 0
FILLED_SEARCH_STATE = 1
RESULTS_STATE = 2

class WebAgentDreamDOMEnv(gym.Env):

    """Gym environment for HTML mode of WebShop environment"""

    def __init__(self, observation_mode='html', window_height: int = 540, window_width: int = 960, scroll_amount: int = 180, scroll_time: int = 150, **kwargs):
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
        
        self._cur_state = None
        self.page_source = None

    def clean_url(self, url: str) -> str:
        """Make a url nicely readable by adding ellipses in long segments"""
        parts = url.split('/')
        parts = [p[:10] + '...' + p[-10:] if len(p) > 20 else p for p in parts]
        return '/'.join(parts)

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
        urls = self.get_available_click_actions()
        
        if action == 0:
            # done = True
            pass
        elif action > len(urls):
            pass
        else:
            with app.test_client() as c:
                self.page_source = c.get(urls[action - 1]).data.decode('utf-8')
                self._cur_state = self.clean_url(urls[action - 1])
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

    @property
    def action_space(self):
        # Recommended to use `get_available_actions` instead
        return NotImplementedError

    @property
    def observation_space(self):
        return NotImplementedError

    def reset(self, seed=None):
        """Create a new session and reset environment variables"""
        if seed is not None:
            self.session = f"fixed_{seed}"
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        
        with app.test_client() as c:
            self.page_source = c.get(f'/{self.session}').data.decode('utf-8')
            self._cur_state = self.clean_url(f'/{self.session}')

        self.instruction_text = self.get_instruction_text()
        self.best_products = self.get_best_products()

        return self.state, None

    def render(self, mode='human'):
        # TODO: Render observation in terminal or WebShop website
        
        # Create a white PIL image and place the self._curstate url text on it, taking account of overflow
        return self.screenshot


    def close(self):
        # TODO: When DB used instead of JSONs, tear down DB here
        pass

    @property
    def screenshot(self):
        img = Image.new('RGB', (self.WINDOW_WIDTH, self.WINDOW_HEIGHT), color = (255, 255, 255))
        """fnt = ImageFont.truetype('arial.ttf', 20)
        d = ImageDraw.Draw(img)
        d.text((10,10), self._cur_state, font=fnt, fill=(0,0,0))

        # Now add page source in tiny font
        fnt = ImageFont.truetype('arial.ttf', 10)
        relevant_content = self.page_source.split("<body>")[1]
        relevant_content = "\n".join(relevant_content.split("\n")[:30])
        d.text((10,30), self.page_source.split("<body>")[1], font=fnt, fill=(0,0,0))"""
        return img

def tag_visible(element):
    """Helper method to strip HTML block of extraneous tags"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and str(element.parent.parent.get("style")) != "display: none;" and not isinstance(element, Comment)
    )
