
square_100 = [x**2 for x in range(101)]
print("All squares from 0 to 100:\n{}".format(square_100), end="\n\n")

square_100_bonus = [x**2 for x in range(101) if x**2 % 2 == 0]
print("All even squares from 0 to 100:\n{}".format(square_100), end="\n\n")
