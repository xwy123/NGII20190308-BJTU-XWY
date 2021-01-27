#测试温度影响
import math
value = [0.001,0.003,0.8,0.001,0.002,0.003,0.006,0.002,0.001,0.001]
value_soft =[]
all_value = 0

for i in value:
    all_value += math.exp(i/5)

for j in value:
    value_soft.append(math.exp(j/5)/all_value)

print(all_value)
print(value_soft)