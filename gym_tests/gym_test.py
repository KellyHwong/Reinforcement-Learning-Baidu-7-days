#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jul-09-20 01:18
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import gym


def main():
    game_name = "FrozenLake-v0"
    env = gym.make(game_name)
    obs = env.reset()
    # env.render()
    # print(obs)
    # print(env.observation_space)
    print(env.action_space)


if __name__ == "__main__":
    main()
