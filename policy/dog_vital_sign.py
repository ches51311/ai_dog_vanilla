import math
import numpy as np

class DogVitalSign:
    def __init__(self) -> None:
        self.reborn_time = 100
        self.reborn_cnt = 0
        self.born()

    def set_default_vital(self):
        # dog's hungry point & mood point
        self.hp = 100.
        self.mp = 100.
        self.life = 0.
        self.prev_hp = self.hp
        self.prev_mp = self.mp
        self.prev_life = self.life
    
    def born(self):
        self.set_default_vital()
        self.state = "just_born"

    def reborn(self):
        self.set_default_vital()
        self.state = "just_reborn"
    
    def update_state(self):
        # just_born/just_reborn/alive/just_death/death
        if self.state in ["just_death", "death"]:
            self.reborn_cnt += 1
        if self.state in ["just_born", "just_reborn"] and self.hp >= 0:
            self.state = "alive"
        elif self.state == "alive" and self.hp < 0:
            self.state = "just_death"
        elif self.state == "just_death" and self.hp < 0:
            self.state = "death"
        elif self.state == "death" and self.reborn_cnt == self.reborn_time:
            self.reborn()
            self.reborn_cnt = 0

    def my_turn(self, obs):
        if self.state not in ["just_death", "death"]:
            self.metabolism(obs)
        self.update_state()

    def metabolism(self, obs):
        self.prev_hp = self.hp
        self.prev_mp = self.mp
        self.prev_life = self.life

        distance = math.dist(obs["breeder_site"][:2], obs["dog_site"][:2])
        # breeder feed add hp
        if distance < 1 and obs["breeder_action"]["feed"] == True:
            print("feed!!")
            self.hp = min(200, self.hp + 20)
            # self.hp += 20
        # breeder touch add mp
        if distance < 1 and obs["breeder_action"]["touch"] == True:
            print("touch!!")
            self.mp = min(200, self.mp + 20)
            # self.mp += 20
        # # breeder near add mp
        # if distance < 0.5:
        #     print("near!!")
        #     self.mp += 5
        # if distance > 5:
        #     self.mp -= 1
        # dog's action influence hp/mp
        dog_momentum = pow(pow(obs["dog_action"]["move"][0], 2) + \
                           pow(obs["dog_action"]["move"][1], 2), 1/2) + abs(obs["dog_action"]["move"][2])
        self.hp -= dog_momentum * 0.01
        self.hp -= abs(obs["dog_action"]["bark"]) * 0.05
        self.hp -= abs(obs["dog_action"]["shake"]) * 0.05
        self.life += 1
        self.hp -= 0.01
        self.mp -= 0.1
        self.mp = max(0., self.mp)
