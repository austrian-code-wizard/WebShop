from gym.envs.registration import register

from web_agent_site.envs.web_agent_dream_env import WebAgentDreamEnv

register(
  id='WebAgentDreamEnv-v0',
  entry_point='web_agent_site.envs:WebAgentDreamEnv',
)