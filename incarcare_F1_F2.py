import pickle
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

with open("F1_12x12", 'rb') as f:
    F1 = pickle.load(f) # lista de 18496 vectori de dim 64

with open("F2_12x12", 'rb') as f:
    F2 = pickle.load(f)


dist_euclid_list = []

for i in range(len(F1)):
    dist = 0.0
    for j in range(len(F1[i])):
        dist += (F1[i][j] - F2[i][j])**2
    dist_euclid_list.append(sqrt(dist))

print(dist_euclid_list)
print(len(dist_euclid_list))
print("min ", min(dist_euclid_list))
print("max ", max(dist_euclid_list))
print(sum(dist_euclid_list)/len(dist_euclid_list))
print(len([x for x in dist_euclid_list if x < 9.0]))

# afisare histo pe dist euclidiene pe intervale
# formare imagine care sa contina distantele - distantele/valorile calculate - 136x136 matrice/imagine - eventual -> rescalare intre 0 si 1

A = [round(x) for x in dist_euclid_list]

min_value = round(min(dist_euclid_list))
max_val = round(max(dist_euclid_list))

# print(type(min_value))

# x = range(min_value, max_val+2)
# y = []
# for i in x:
#     y.append(0)
#
# for value in dist_euclid_list:
#     for i in range(len(x)):
#         if float(x[i]) < value and value > float(x[i]):
#             y[i] += 1
#
# y = [x/len(dist_euclid_list) for x in y]
# print(y)
# # plt.hist(x, y)
# # plt.show()

plt.hist(dist_euclid_list, bins=50)
plt.show()
dim = int(sqrt(len(dist_euclid_list)))
#print(dim)
new_list = np.array(dist_euclid_list)
new_image = np.reshape(new_list, (dim, dim))
print(new_image)

plt.imshow(new_image)
plt.show()




