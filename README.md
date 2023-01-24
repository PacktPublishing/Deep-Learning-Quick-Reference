


# Deep Learning Quick Reference
This is the code repository for [Deep Learning Quick Reference](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-quick-reference?utm_source=github&utm_medium=repository&utm_campaign=9781788837996), published by [Packt](https://www.packtpub.com/?utm_source=github). It contains all the supporting project files necessary to work through the book from start to finish.
## About the Book
Deep learning has become an essential necessity to enter the world of artificial intelligence. With this book deep learning techniques will become more accessible, practical, and relevant to practicing data scientists. It moves deep learning from academia to the real world through practical examples.

You will learn how Tensor Board is used to monitor the training of deep neural networks and solve binary classification problems using deep learning. Readers will then learn to optimize hyperparameters in their deep learning models. The book then takes the readers through the practical implementation of training CNN's, RNN's, and LSTM's with word embeddings and seq2seq models from scratch. Later the book explores advanced topics such as Deep Q Network to solve an autonomous agent problem and how to use two adversarial networks to generate artificial images that appear real. For implementation purposes, we look at popular Python-based deep learning frameworks such as Keras and Tensorflow, Each chapter provides best practices and safe choices to help readers make the right decision while training deep neural networks.

By the end of this book, you will be able to solve real-world problems quickly with deep neural networks.

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with a number followed by the application name. For example, Chapter02.



The code will look like the following:
```
a = tf.constant([1.0,</span> 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3],
name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
```

1. I assume that you're already experienced with more traditional data science and
predictive modeling techniques such as Linear/Logistic Regression and Random
Forest. If this is your first experience with machine learning, this may be a little
difficult for you.
2. I also assume that you have at least some experience in programming with
Python, or at least another programming language such as Java or C++.
3. Deep learning is computationally intensive, and some of the models we build
here require an NVIDIA GPU to run in a reasonable amount of time. If you don't
own a fast GPU, you may wish to use a GPU-based cloud instance on either
Amazon Web Services or Google Cloud Platform.

## Related Products
* [Artificial Intelligence and Deep Learning for Decision Makers](https://www.packtpub.com/big-data-and-business-intelligence/artificial-intelligence-and-deep-learning-decision-makers?utm_source=github&utm_medium=repository&utm_campaign=9781788294652)

* [Apache Spark Deep Learning Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/apache-spark-deep-learning-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788474221)

* [Keras Deep Learning Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/keras-deep-learning-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788621755)
### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781788837996">https://packt.link/free-ebook/9781788837996 </a> </p>