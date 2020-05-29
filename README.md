# Smoothness of Functions Learned by Neural Networks

Code of my internship project at IST Austria,
which was then used as my Bachelor thesis.
The internship took place between February and April 2020.

## Abstract

Modern neural networks can easily fit their training set perfectly.
Surprisingly, they generalize well despite being "overfit" in this way,
defying the bias--variance trade-off. A prevalent explanation is that
stochastic gradient descent has an implicit bias which leads it to learn
functions which are simple, and these simple functions generalize well.
However, the specifics of this implicit bias are not well understood. In this
work, we explore the hypothesis that SGD is implicitly biased towards learning
functions which are smooth. We propose several measures to formalize the
intuitive notion of smoothness, and conduct experiments to determine whether
they are implicitly being optimized for. We exclude the possibility that
smoothness measures based on the first derivative (the gradient) are being
implicitly optimized for. Measures based on the second derivative (the
Hessian), on the other hand, show promising results.
