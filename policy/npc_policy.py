import numpy as np
import math

class NPCPolicy:
    def __init__(self):
        self.feed_cooldown = 0
        self.touch_cooldown = 0
        self.walk_cooldown = 0
    def cnt_to_0(self, cnt):
        return cnt - 1 if cnt > 0 else cnt
    def walk(self, man_site):
        if self.walk_cooldown == 0:
            r = np.random.uniform(-0.2,0.2,3)
            if man_site[0] > 2:
                r[0] = abs(r[0]) * -1
            if man_site[0] < 0:
                r[0] = abs(r[0])
            if man_site[1] > 2:
                r[1] = abs(r[1]) * -1
            if man_site[1] < 0:
                r[1] = abs(r[1])
            self.walk_cooldown = 300
            self.move = r
        self.walk_cooldown = self.cnt_to_0(self.walk_cooldown)
        return self.move
    def my_turn(self, obs):
        distance = math.dist(obs["man_site"][:2], obs["dog_site"][:2])
        action = {"move": self.walk(obs["man_site"]), "feed": False, "touch": False}
        if distance < 0.3 and self.feed_cooldown <= 0 and \
           obs["dog_action"]["bark"] > 0.5:
            action["feed"] = True
            self.feed_cooldown = 100
        if distance < 0.3 and self.touch_cooldown <= 0 and \
            obs["dog_action"]["shake"] > 0.5:
            action["touch"] = True
            self.touch_cooldown = 100
        self.feed_cooldown = self.cnt_to_0(self.feed_cooldown)
        self.touch_cooldown = self.cnt_to_0(self.touch_cooldown)
        obs["man_action"] = action
