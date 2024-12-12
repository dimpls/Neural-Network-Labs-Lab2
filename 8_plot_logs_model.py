import matplotlib.pyplot as plt

# Данные для графиков (первый запуск - Run 0)
epochs_0 = [1, 2, 3, 4, 5]
eval_accuracy_0 = [0.5333333333333333, 0.5787878787878787, 0.6189393939393939, 0.6257575757575757, 0.6303030303030303]
eval_f1_0 = [0.5307956090255949, 0.5783702069027907, 0.6192121485804862, 0.6258691173242108, 0.6299650407971975]
eval_precision_0 = [0.5377111205594873, 0.5803814064288423, 0.6208463242370547, 0.6279857808892619, 0.6307854006528139]
eval_recall_0 = [0.5333333333333333, 0.5787878787878787, 0.6189393939393939, 0.6257575757575757, 0.6303030303030303]

# Данные для графиков (второй запуск - Run 1)
epochs_1 = [1, 2, 3]
eval_accuracy_1 = [0.7484848484848485, 0.7962121212121213, 0.8090909090909091]
eval_f1_1 = [0.7496347537639011, 0.7947991087362902, 0.8084305271609951]
eval_precision_1 = [0.7604705370653905, 0.798892632322947, 0.8092844936846206]
eval_recall_1 = [0.7484848484848485, 0.7962121212121213, 0.8090909090909091]

# Данные для графиков (третий запуск - Run 2)
epochs_2 = [1, 2, 3, 4, 5]
eval_accuracy_2 = [0.7507575757575757, 0.7924242424242425, 0.8083333333333333, 0.8083333333333333, 0.8053030303030303]
eval_f1_2 = [0.7519902256914587, 0.7894902415265258, 0.8069901228212695, 0.8076994014032538, 0.8050740289239843]
eval_precision_2 = [0.7658063125034876, 0.8000878350939984, 0.8105596530402542, 0.8087244360931758, 0.8050907116217071]
eval_recall_2 = [0.7507575757575757, 0.7924242424242425, 0.8083333333333333, 0.8083333333333333, 0.8053030303030303]

# Данные для графиков (четвертый запуск - Run 3)
epochs_3 = [1, 2, 3, 4, 5]
eval_accuracy_3 = [0.7545454545454545, 0.7795454545454545, 0.7886363636363637, 0.803030303030303, 0.8121212121212121]
eval_f1_3 = [0.7546691348307509, 0.7770991361247797, 0.7881136591611841, 0.8023836407409924, 0.8117539114719221]
eval_precision_3 = [0.7559281221544984, 0.7863645338946534, 0.7899677312793182, 0.8043399316160262, 0.8121922454480329]
eval_recall_3 = [0.7545454545454545, 0.7795454545454545, 0.7886363636363637, 0.803030303030303, 0.8121212121212121]

# Данные для графиков (пятый запуск - Run 17)
epochs_17 = [1, 2, 3, 4]
eval_accuracy_17 = [0.7628787878787879, 0.7833333333333333, 0.8083333333333333, 0.8045454545454546]
eval_f1_17 = [0.7632168264400775, 0.7811156252892308, 0.8074840361750052, 0.8041019333189883]
eval_precision_17 = [0.7664184880328457, 0.7934003842816572, 0.8098346311862136, 0.8054086561390654]
eval_recall_17 = [0.7628787878787879, 0.7833333333333333, 0.8083333333333333, 0.8045454545454546]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(epochs_0, eval_accuracy_0, marker='o', color='b', label='Запуск 0')
axs[0, 0].plot(epochs_1, eval_accuracy_1, marker='o', color='orange', label='Запуск 1')
axs[0, 0].plot(epochs_2, eval_accuracy_2, marker='o', color='green', label='Запуск 2')
axs[0, 0].plot(epochs_3, eval_accuracy_3, marker='o', color='purple', label='Запуск 3')
axs[0, 0].plot(epochs_17, eval_accuracy_17, marker='o', color='red', label='Запуск 17')
axs[0, 0].set_title('Точность (Accuracy) по эпохам')
axs[0, 0].set_xlabel('Эпоха')
axs[0, 0].set_ylabel('Точность (Accuracy)')
axs[0, 0].legend()

axs[0, 1].plot(epochs_0, eval_f1_0, marker='o', color='g', label='Запуск 0')
axs[0, 1].plot(epochs_1, eval_f1_1, marker='o', color='orange', label='Запуск 1')
axs[0, 1].plot(epochs_2, eval_f1_2, marker='o', color='green', label='Запуск 2')
axs[0, 1].plot(epochs_3, eval_f1_3, marker='o', color='purple', label='Запуск 3')
axs[0, 1].plot(epochs_17, eval_f1_17, marker='o', color='red', label='Запуск 17')
axs[0, 1].set_title('F1-мера по эпохам')
axs[0, 1].set_xlabel('Эпоха')
axs[0, 1].set_ylabel('F1-мера')
axs[0, 1].legend()

axs[1, 0].plot(epochs_0, eval_precision_0, marker='o', color='b', label='Запуск 0')
axs[1, 0].plot(epochs_1, eval_precision_1, marker='o', color='orange', label='Запуск 1')
axs[1, 0].plot(epochs_2, eval_precision_2, marker='o', color='green', label='Запуск 2')
axs[1, 0].plot(epochs_3, eval_precision_3, marker='o', color='purple', label='Запуск 3')
axs[1, 0].plot(epochs_17, eval_precision_17, marker='o', color='red', label='Запуск 17')
axs[1, 0].set_title('Точность (Precision) по эпохам')
axs[1, 0].set_xlabel('Эпоха')
axs[1, 0].set_ylabel('Точность (Precision)')
axs[1, 0].legend()

axs[1, 1].plot(epochs_0, eval_recall_0, marker='o', color='b', label='Запуск 0')
axs[1, 1].plot(epochs_1, eval_recall_1, marker='o', color='orange', label='Запуск 1')
axs[1, 1].plot(epochs_2, eval_recall_2, marker='o', color='green', label='Запуск 2')
axs[1, 1].plot(epochs_3, eval_recall_3, marker='o', color='purple', label='Запуск 3')
axs[1, 1].plot(epochs_17, eval_recall_17, marker='o', color='red', label='Запуск 17')
axs[1, 1].set_title('Полнота (Recall) по эпохам')
axs[1, 1].set_xlabel('Эпоха')
axs[1, 1].set_ylabel('Полнота (Recall)')
axs[1, 1].legend()

plt.tight_layout()
plt.show()
