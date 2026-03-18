import random

random_numbers = [random.random() for _ in range(100)]

random_numbers.sort()

print("정렬된 숫자 100개:")
print(random_numbers)
