class Policy:
    def __init__(self,bot_x, bot_y, crew_x, crew_y, direction):
        self.bot_x = bot_x
        self.bot_y = bot_y
        self.crew_x = crew_x
        self.crew_y = crew_y
        self.direction = direction

    def get_dict(self):
        return {
            'BOT_X': self.bot_x,
            'BOT_Y': self.bot_y,
            'CREW_X': self.crew_x,
            'CREW_Y': self.crew_y,
            'Optimal_Direction': self.direction
        }