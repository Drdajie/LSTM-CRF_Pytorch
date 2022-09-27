import torch

def log_sum_exp_matrix(matrix):
    """
    将矩阵的列向量进行 log sum exp 操作。
    """
    max_score = matrix[range(matrix.shape[0]), torch.argmax(matrix, dim=1)]
    return max_score + \
        torch.log(torch.sum(torch.exp(matrix - max_score.view(-1, 1)), dim=1, keepdim=True)).view(1, -1)

a = torch.randn(2, 2)
print(a)
print(torch.logsumexp(a, dim=1))
print(log_sum_exp_matrix(a))
print(torch.log(torch.exp(a[0][0]) + torch.exp(a[0][1])))
print(torch.log(torch.exp(a[1][0]) + torch.exp(a[1][1])))