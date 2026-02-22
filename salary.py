from bonus import bonus
from total import total
basic = float(input("Enter basic salary: "))
bon = bonus(basic)
tot = total(basic, bon)
print("Bonus =", bon)
print("Total Salary =", tot)

