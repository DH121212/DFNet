import utility
import data
import model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if args.cuda else 'cpu')
    utility.set_seed(args.seed)

    checkpoint = Checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        model = model.to(device)

        # ========== 新增：计算参数量和 FLOPs ==========
        from thop import profile

        # 构造一个示例输入：假设输入是 3 通道、224x224 的 LR 图像
        # 注意：根据你的模型输入要求调整通道数（如红外单通道则为 1）
        input_channels = 3  # 如果是红外单通道图像，请改为 1
        dummy_input = torch.randn(1, input_channels, 224, 224).to(device)

        # 计算 FLOPs 和 Params
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        # 转换为更易读的单位
        from thop import clever_format

        flops_str, params_str = clever_format([flops, params], '%.3f')

        print(f"Model Parameters: {params_str} ({params:.0f})")
        print(f"FLOPs (224×224 input): {flops_str} ({flops:.0f})")
        # ==============================================

        if not args.test_only:
            loss_instance = loss.Loss(args, checkpoint)
            loss_instance = loss_instance.to(device)
        else:
            loss_instance = None

        t = Trainer(args, loader, model, loss_instance, checkpoint)

        while not t.terminate():
            t.train()
            t.test()
            checkpoint.done()

#
# if __name__ == '__main__':
#     device = torch.device('cuda' if args.cuda else 'cpu')
#     utility.set_seed(args.seed)
#
#     checkpoint = Checkpoint(args)
#     if checkpoint.ok:
#         loader = data.Data(args)
#         model = model.Model(args, checkpoint)
#
#         # ========== 关键修改：将 model 和 loss 移到 device ==========
#         model = model.to(device)
#
#         if not args.test_only:
#             loss_instance = loss.Loss(args, checkpoint)
#             loss_instance = loss_instance.to(device)  # ←←← 必须移到 GPU！
#         else:
#             loss_instance = None
#
#         # 传入 device 给 Trainer（确保内部也一致）
#         t = Trainer(args, loader, model, loss_instance, checkpoint)
#
#         while not t.terminate():
#             t.train()
#             t.test()
#             checkpoint.done()
#     # utility.set_seed(args.seed)
#     # checkpoint = Checkpoint(args)
#     # if checkpoint.ok:
#     #     loader = data.Data(args)
#     #     model = model.Model(args, checkpoint)
#     #     loss = loss.Loss(args, checkpoint) if not args.test_only else None
#     #     t = Trainer(args, loader, model, loss, checkpoint)
#     #     while not t.terminate():
#     #         t.train()
#     #         t.test()
#     #         checkpoint.done()

