import torch
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    root_path = 'result/ResNet50_pretrain_stage1'
    data = torch.load(os.path.join(root_path, 'train_result.pt'))
    test_data = torch.load(os.path.join(root_path, 'test_result.pt'))


    show_attr = 'top_1'
    test_rate = 5
    loss = [d[show_attr] for d in data]
    test_loss = [d[show_attr] for d in test_data]
    test_x = torch.arange(1, len(loss)+1, test_rate)

    
    plt.title(f'{show_attr} curve')
    plt.plot(loss, 'o-', linewidth=2, label='train_loss')
    plt.plot(test_x.tolist(), test_loss, 'o-', linewidth=2, label='test_loss')
    plt.legend()
    plt.grid()
    plt.show()
