def meow():
    """Generate Generator which doubels 'Meow' each iteration."""
    num = 1
    while True:
        yield ("Meow " * num).strip()
        num *= 2


meoww = meow()
for _ in range(10):
    print(next(meoww), end="\n\n")
