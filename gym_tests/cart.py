#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-09-20 01:24
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
