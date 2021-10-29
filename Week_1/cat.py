class Cat():
    """Create a cat with a name"""

    def __init__(self, name):
        self.name = name

    def greet(self, cat_to_greet):
        """Introduces yourself and greet another cat"""
        print("\nHello I am {0}!\nI see you are also a cool fluffy kitty {1},\nlet’s together purr at the human, so that they shall give us food.\n\n"
              .format(self.name, cat_to_greet.get_name()))

    def get_name(self):
        """Retruns the name of a cat"""
        return self.name
