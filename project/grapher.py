import pandas as pd
import matplotlib.pyplot as plt

csv_files = ['model_4_0.csv', 'model_4_1.csv', 'model_4_2.csv', 'model_4_3.csv', 'model_4_4.csv']

colors = plt.cm.get_cmap('tab10', len(csv_files))

epochs = range(1, 26)
i = 0
for file in csv_files:
    i += 1
    train_losses = []
    test_losses = []
    text = open(file, "r").read()
    lines = text.split('\n')
    for line in lines:
        line = line.split(',')
        if len(line) < 3:
            continue
        train_losses.append(float(line[1]))
        test_losses.append(float(line[2]))
    color = colors(i-1)
    plt.plot(epochs, train_losses, label=f'Train Loss Model {i}', linestyle='--', color=color)
    plt.plot(epochs, test_losses, label=f'Test Loss Model {i}', color=color)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss over Epochs for 1 pooling layers')
#plt.legend(loc='best')
plt.grid(True)
plt.show()
