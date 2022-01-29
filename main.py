import torch
import matplotlib.pyplot as plt
from arima_model import *

def train(input = None,epochs=1000):
    if input==None:
        input,errors_vec = create_random_arima(num_samples=20)
        # plt.plot(input.squeeze().detach().numpy())
    train = input[:,0:14]
    test_error = errors_vec[:,14:]
    #W is the diff vector in order to get rid off the integration variable X_{t-1},
    # the parameters to calculate stay the same
    W = torch.diff(train)
    #Task 1 a
    model = Arima_model(train.shape[1])
    learning_rate = 0.001
    optimizer = torch.optim.SGD([model.theta,model.drift], lr=learning_rate,momentum=0.9)
    loss_list =[]
    #Task 1 b
    for i in range(epochs):
        result = model(W)
        output = loss_from_max_likelihood(result)
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(output.detach())
    #Loss curve
    # plt.plot(range(len(loss_list)), loss_list)
    # plt.show()
    print(model.theta, model.drift)
    #Task 1 c
    sig_sqr = model.calculate_sigma_sqr(output).detach().numpy()
    prob = probabilty_estimator(test_error, sig_sqr)
    print(prob)

if __name__ == '__main__':
    #Solution for task 1
    train()

#Solution for task 2
# If I had to predict the middle of the signal I would predict from the first 7 samples and then I would take
# the last 7 samples reverse it and try to predict the previous 7 samples as future data. The model with the
# best MAE error (or another metric) will be taken. A mix between models could also work if possible