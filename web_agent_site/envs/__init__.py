from gym.envs.registration import register

from web_agent_site.envs.web_agent_site_env import WebAgentSiteEnv
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from web_agent_site.envs.web_agent_dream_env import WebAgentDreamEnv

register(
  id='WebAgentSiteEnv-v0',
  entry_point='web_agent_site.envs:WebAgentSiteEnv',
)

register(
  id='WebAgentTextEnv-v0',
  entry_point='web_agent_site.envs:WebAgentTextEnv',
)

register(
  id='WebAgentDreamEnv-v0',
  entry_point='web_agent_site.envs:WebAgentDreamEnv',
)