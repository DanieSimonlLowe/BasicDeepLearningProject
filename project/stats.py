import scipy.stats as stats

# Example datasets (replace with your actual datasets)
#
csv_file1 = ['model_1_3.csv', 'model_1_4.csv', 'model_1_5.csv', 'model_1_6.csv', 'model_1_7.csv']
csv_file2 = ['model_2.csv', 'model_2_2.csv', 'model_2_3.csv', 'model_2_4.csv', 'model_2_5.csv']
csv_file3 = ['model_3_1.csv', 'model_3_2.csv', 'model_3_3.csv', 'model_3_4.csv', 'model_3_5.csv']
csv_file4 = ['model_4_0.csv', 'model_4_1.csv', 'model_4_2.csv', 'model_4_3.csv', 'model_4_4.csv']
csv_file5 = ['model_5_1.csv', 'model_5_2.csv', 'model_5_3.csv', 'model_5_4.csv', 'model_5_5.csv']

files = [csv_file1,csv_file3,csv_file4,csv_file5,csv_file2,]

def get_stat(csv_file1,csv_file2):
    data1 = []
    data2 = []

    for file in csv_file1:
        text = open(file, "r").read()
        lines = text.split('\n')
        for line in lines:
            line = line.split(',')
            if len(line) < 3:
                continue
            data1.append(float(line[2]))

    for file in csv_file2:
        text = open(file, "r").read()
        lines = text.split('\n')
        for line in lines:
            line = line.split(',')
            if len(line) < 3:
                continue
            data2.append(float(line[2]))
        

    # Perform the Mann-Whitney U Test
    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    return stat, p_value


out = ''

for i in range(5):
    out += f'{i} pooling layers&'
    for j in range(5):
        value, _ = get_stat(files[i],files[j])
        out += f'{value:.3}&'
    out = out[:-1] + '\\\\ \n \\hline \n'

print(out)