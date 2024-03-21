import numpy as np

def power_mean(vals, p):
    """
    :param vals: values to compute the pmean on
    :param p: p value
    :return: Generalized mean
    """
    vals_np = np.array(vals)
    # Ensure positive numbers
    if vals_np.min() <= 0:
        vals_np += np.abs(vals_np.min()) + 0.000000001
    else:
        vals_np += 0.000000001

    if p == 0:
        # defined as the geometric mean for 0
        return (np.prod(vals_np)**(1/len(vals_np)))

    return (np.mean(vals_np**p)**(1 / p)).real

if __name__ == '__main__':
    print(power_mean([0.1, 0.3, 0.2, 0.15, 0.2, 0.1], 1))
    print(sum([0.1, 0.3, 0.2, 0.15, 0.2, 0.1])/6)

    print(power_mean([0.1, 0.3, 0.2, 0.15, 0.2, 0.1], 300))
    print(max([0.1, 0.3, 0.2, 0.15, 0.2, 0.1]))

    print(power_mean([0.1, 0.3, 0.2, 0.15, 0.2, 0.1], -300))
    print(min([0.1, 0.3, 0.2, 0.15, 0.2, 0.1]))

    print(power_mean([0.1, 0.3, 0.2, 0.15, 0.2, 0.1], 2))
    print(np.sqrt(sum([i ** 2 for i in [0.1, 0.3, 0.2, 0.15, 0.2, 0.1]]) / 6))

    print(power_mean([0.1, 0.3, 0.2, 0.15, 0.2, 0.1], -1))
    print(6 / sum([1/i for i in [0.1, 0.3, 0.2, 0.15, 0.2, 0.1]]))
