from matplotlib import pyplot as plt


y_data = [5, 5, 5, 5, 3, 3, 3, 3]

e = 8
x_data = list(range(int(e)))

y_sums = []
y_averages = []
x_sums_averages = list(range(int(e/4)))

bin_size = 4
bins = int(e/bin_size)

for i in range(bins):
    part = y_data[(i*bin_size): ((i*bin_size)+bin_size)]
    sum_part = sum(part)
    y_sums.append(sum_part)
    average_part = sum_part / len(part)
    y_averages.append(average_part)


fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax1.scatter(x_data, y_data, c='r', label='data')
ax1.set_xlabel("episode nr.")
ax1.set_ylabel("total reward")
ax1.set_title('total reward per episode')

ax2.scatter(x_sums_averages, y_sums, c='r', label='data')
ax2.set_xlabel("batch nr.")
ax2.set_ylabel("sum reward")
ax2.set_title('total reward per batch of size 4')

ax3.scatter(x_sums_averages, y_averages, c='r', label='data')
ax3.set_xlabel("batch nr.")
ax3.set_ylabel("sum reward")
ax3.set_title('total reward per batch of size 4')

fig.tight_layout()

fig.savefig("test.png")
