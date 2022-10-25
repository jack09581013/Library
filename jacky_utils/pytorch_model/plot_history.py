import matplotlib.pyplot as plt
import numpy as np
import os
from colorama import Fore, Style
from config import Config


def get_config():
    return Config().set_mode('plot_history')


def get_color(value):
    if value < 0:
        return Fore.GREEN
    else:
        return Fore.RED


def trend_regression(loss_trend, method='corr'):
    """Loss descent checking"""
    if method == 'regression':
        b = loss_trend.reshape(-1, 1)  # b: (n, 1)
        A = np.concatenate(
            [np.arange(len(b)).reshape(-1, 1), np.ones((len(b), 1))], axis=1)
        x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
        return float(x[0, 0])

    elif method == 'corr':
        A = loss_trend
        B = np.arange(len(A))
        corr = np.corrcoef(A, B)[0, 1]
        return corr

    else:
        raise Exception(f'Method "{method}" is not valid')


config = get_config()
trend_kernel = 1  # version (plot) + trend kernel = real model version, trend_kernel = [1, n]
trend_regression_size = 1  # to see the loss is decent or not, trend_regression_size = [1, n]
trend_method = ['corr', 'regression'][1]
start_version = config.fine_tune_version  # start_version = [1, n], real_version = x + start_version

version, loss_history = config.profile.load_history(config.version)

version_loss_list = []
for i, loss in enumerate(loss_history["test"][start_version - 1:]):
    version_loss_list.append((i + start_version, loss))
version_loss_list.sort(key=lambda x: x[1])
print(f'{Fore.GREEN}Top 5 best versions: {[x[0] for x in version_loss_list[:5]]}{Style.RESET_ALL}')
print(f'{Fore.GREEN}Top 5 best loss: {np.array([x[1] for x in version_loss_list[:5]])}{Style.RESET_ALL}')

print(f'Model name: {config.profile}')
print('Number of epochs:', len(loss_history['test']))
print('Trend kernel size:', trend_kernel)
print('Trend regression size:', trend_regression_size)

if len(loss_history['test']) == 1:
    marker = 'o'
else:
    marker = 'o'

assert start_version >= 1 and isinstance(start_version, int)
train_loss_history = loss_history['train'][start_version - 1:]
test_loss_history = loss_history['test'][start_version - 1:]
print('Size of loss history:', len(train_loss_history))

if config.save_image:
    plt.figure(figsize=(16, 10))
else:
    plt.figure()

plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

p_train = plt.plot(train_loss_history[(trend_kernel - 1):], label='Train', marker=marker)
p_test = plt.plot(test_loss_history[(trend_kernel - 1):], label='Test', marker=marker)

if trend_kernel > 1:
    test_loss_trend = np.array(test_loss_history)
    test_loss_trend = np.convolve(test_loss_trend, np.ones(trend_kernel, )) / trend_kernel
    test_loss_trend = test_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
    # plt.plot(test_loss_trend, label='Test Trend', marker=marker, color=p_train[0].get_color(), linestyle='--')
    plt.plot(test_loss_trend, label='Test Trend', marker=marker)

    train_loss_trend = np.array(train_loss_history)
    train_loss_trend = np.convolve(train_loss_trend, np.ones(trend_kernel, )) / trend_kernel
    train_loss_trend = train_loss_trend[(trend_kernel - 1):-trend_kernel + 1]
    # plt.plot(train_loss_trend, label='Train Trend', marker=marker, color=p_test[0].get_color(), linestyle='--')
    plt.plot(train_loss_trend, label='Train Trend', marker=marker)

    print('Trend method:', trend_method)

    train_loss_trend_regression = trend_regression(train_loss_trend[-trend_regression_size:], method=trend_method)
    test_loss_trend_regression = trend_regression(test_loss_trend[-trend_regression_size:], method=trend_method)
    test_train_diff = test_loss_trend[-1] - train_loss_trend[-1]
    train_loss_trend_color = None
    test_loss_trend_color = None
    test_train_diff_color = None

    print(f'Train loss trend: {get_color(train_loss_trend_regression)}{train_loss_trend_regression:.2e}{Style.RESET_ALL}')
    print(f'Test loss trend: {get_color(test_loss_trend_regression)}{test_loss_trend_regression:.2e}{Style.RESET_ALL}')
    print(f'Last test loss - train loss (large is overfitting): {get_color(test_train_diff)}{test_train_diff:.2e}{Style.RESET_ALL}')

# plt.axhline(0, color='k', linestyle='--', label='x-axis')
# plt.axhline(baseline_loss, color='r', linestyle='--', label='Baseline loss')

plt.legend()
if config.save_image:
    os.makedirs('result', exist_ok=True)
    plt.savefig(os.path.join('result', f'history-{str(config.profile)}-{version}.png'))
else:
    plt.show()
plt.close()