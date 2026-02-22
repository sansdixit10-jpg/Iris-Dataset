def calculate_bonus(basic_salary):
    return basic_salary * 0.10   # 10% bonus

def calculate_total_salary(basic_salary, bonus):
    return basic_salary + bonus

basic = float(input("Enter basic salary: "))

bonus = calculate_bonus(basic)
total = calculate_total_salary(basic, bonus)

print("Bonus =", bonus)
print("Total Salary =", total)
print("Hello from monolith.py!")