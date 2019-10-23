# movieReview
DeepLearning With Python Chapter 3 -- movieReview

## 导入数据集
    from keras.datasets import imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
### 查看第一条数据
    pritn(train_data[0])

## 将评论从索引翻译为文字
### word_index is a dictionary mapping words to an integer index
    word_index = imdb.get_word_index()
### We reverse it, mapping integer indices to words
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
### We decode the review; note that our indices were offset by 3
### because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

## 将整数序列编码为二进制矩阵
    import numpy as np 

    def vectorize_sequences(sequence, dimension=10000):
        #Creat an all-zero matrix of shape (len(sequence),dimension)
        results = np.zeros((len(sequence), dimension))
        for i, sequence in enumerate(sequence):
            results[i, sequence] = 1. # set specific indices of results[i] to 1s
        return results

### Our vectorized training data
    x_train = vectorize_sequences(train_data)
### Our vectorized test data
    x_test = vectorize_sequences(test_data)

    print(x_train[0])

### Our vectorized labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

## 模型定义
    from keras import models
    from keras import layers

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

## 配置rmsprop优化器和binary_crossentropy(二元交叉熵)损失函数
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

## 配置自定义优化器的参数
    from keras import optimizers

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    from keras import losses
    from keras import metrics

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

## 留出10000个样本作为验证集
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

## 训练20轮，没批量数据包含512个样本
    history = model.fit(partial_x_train, partial_y_train, epochs = 20, 
                        batch_size = 512, validation_data=(x_val, y_val))

    history_dict = history.history
    history_dict.keys()

## 绘制训练损失和验证损失
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')    #'bo'表示蓝色圆点
    plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')    #'b'表示蓝色实线

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

## 绘制训练和验证精度
    plt.clf()    #清空图像
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


## 从头开始训练一个网络，训练4轮，然后在测试数据上评估模型
    model = models.Sequential()
    model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
    model.add(layers.Dense(16, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop',
                loss = 'binary_crossentropy',
                metrics = ['accuracy'])

    model.fit(x_train, y_train, epochs = 4, batch_size = 512)
    results = model.evaluate(x_test, y_test)

## 使用训练好的网络进行预测
    model.predict(x_test)
    print(y_test)
