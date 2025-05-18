import os
import sys
import json
import argparse
import time
from collections import defaultdict
from easydict import EasyDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torch.cuda.amp as amp   # TODO

from transformers import AutoTokenizer, PretrainedConfig
from transformers import AutoModel

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from utils.distributed import all_gather

from optim import get_lr_sched
from optim.misc import build_optimizer

from parser import load_parser, parse_with_config

from data.loader import MetaLoader, PrefetchLoader, build_dataloader
from data.dataset import R2RTextPathData
from data.tasks import (
    MlmDataset, mlm_collate,
    MrcDataset, mrc_collate,
    SapDataset, sap_collate)

from model.pretrain_cmt import GlocalTextPathCMTPreTraining


def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):
    """
    创建数据加载器
    
    参数：
    - data_cfg: 数据配置
    - nav_db: 导航数据库
    - tok: 分词器
    - is_train: 是否为训练模式
    - device: 设备
    - opts: 命令行参数
    
    返回：
    - dataloaders: 任务数据加载器的字典
    """
    dataloaders = {}
    for k, task_name in enumerate(data_cfg.tasks):
        # 根据任务类型创建相应的数据集
        if task_name == 'mlm':
            # MLM(掩码语言模型)任务 - 预测被掩码的文本单词
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == 'mrc':
            # MRC(掩码区域分类)任务 - 预测被掩码的图像区域
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob, end_vp_pos_ratio=0.2)
            task_collate_fn = mrc_collate
        elif task_name == 'sap':
            # SAP(行为预测)任务 - 预测下一步导航行为
            task_dataset = SapDataset(nav_db, tok, end_vp_pos_ratio=0.2)
            task_collate_fn = sap_collate
        else:
            raise ValueError(f'Undefined task {task_name}')

        LOGGER.info(f"{task_name}: {len(task_dataset)} samples loaded")

        # 创建数据加载器
        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        # 训练模式下，需要记录任务的混合比例
        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders


def main(opts):
    """
    主函数：设置环境，创建模型和数据加载器，执行训练和验证
    
    参数：
    - opts: 命令行参数
    """
    # 设置CUDA环境
    default_gpu, n_gpu, device = set_cuda(opts)

    if default_gpu:
        LOGGER.info(
            'device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}'.format(
                device, n_gpu, bool(opts.local_rank != -1), opts.fp16
            )
        )
 
    # 设置随机种子
    seed = opts.seed
    if opts.local_rank != -1:
        seed += opts.rank
    set_random_seed(seed)

    # 设置日志和模型保存器
    if default_gpu:
        save_training_meta(opts)
        TB_LOGGER.create(os.path.join(opts.output_dir, 'logs'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(os.path.join(opts.output_dir, 'ckpts'))
        add_log_to_file(os.path.join(opts.output_dir, 'logs', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    # 加载模型配置
    # 模型配置，包含预训练任务列表
    model_config = PretrainedConfig.from_json_file(opts.model_config)
    model_config.pretrain_tasks = []
    for train_dataset_config in opts.train_datasets.values():
        model_config.pretrain_tasks.extend(train_dataset_config['tasks'])
    model_config.pretrain_tasks = set(model_config.pretrain_tasks)

    # 创建分词器
    tokenizer = AutoTokenizer.from_pretrained(model_config.lang_bert_name)

    # 准备模型 - 从检查点加载或使用预训练模型初始化
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint, map_location=lambda storage, loc: storage)
    else:
        checkpoint = {}
        if opts.init_pretrained == 'bert':
            # 使用BERT模型初始化
            tmp = AutoModel.from_pretrained(model_config.lang_bert_name)
            for param_name, param in tmp.named_parameters():
                checkpoint[param_name] = param
            if model_config.lang_bert_name == 'xlm-roberta-base':
                # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                # 扩展token_type嵌入以支持图像嵌入
                checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                    [checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                )
            del tmp
        elif opts.init_pretrained == 'lxmert':
            # 使用LXMERT模型初始化（一种多模态预训练模型）
            tmp = torch.load(
                'duet/datasets/pretrained/LXMERT/model_LXRT.pth', 
                map_location=lambda storage, loc: storage
            )
            for param_name, param in tmp.items():
                param_name = param_name.replace('module.', '')
                if 'bert.encoder.layer' in param_name:
                    param_name = param_name.replace('bert.encoder.layer', 'bert.lang_encoder.layer')
                    checkpoint[param_name] = param
                elif 'bert.encoder.x_layers' in param_name:
                    param_name1 = param_name.replace('bert.encoder.x_layers', 'bert.local_encoder.encoder.x_layers')
                    param_name2 = param_name.replace('bert.encoder.x_layers', 'bert.global_encoder.encoder.x_layers')
                    checkpoint[param_name1] = checkpoint[param_name2] = param
                elif 'cls.predictions' in param_name:
                    param_name = param_name.replace('cls.predictions', 'mlm_head.predictions')
                    checkpoint[param_name] = param
                else:
                    checkpoint[param_name] = param
            del tmp
    
    # 定义模型类
    model_class = GlocalTextPathCMTPreTraining
    
    # 创建模型
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, config=model_config, state_dict=checkpoint
    )
    model.train()
    set_dropout(model, opts.dropout)
    model = wrap_model(model, device, opts.local_rank)
    del checkpoint
    
    # 加载训练数据集
    # R2R(Room-to-Room)数据集，用于视觉语言导航
    data_cfg = EasyDict(opts.train_datasets['R2R'])
    # 数据集信息
    # data_cfg.train_traj_files： 存储了轨迹文件路径
    # data_cfg.img_ft_file： 存储了图像特征文件路径
    # data_cfg.scanvp_cands_file： 存储了扫描视图候选文件路径
    # data_cfg.connectivity_dir： 存储了连接性文件路径
    # scanvp_cands_file 数据格式
    #     "scan_id_viewpoint_id": {
    #     "connected_viewpoint_id": [
    #         viewpoint_index,     // Index of the view (0-35) showing this connection
    #         relative_angle_dist, // Distance in terms of relative angle 
    #         relative_heading,    // Relative heading angle to target viewpoint
    #         relative_elevation   // Relative elevation angle to target viewpoint
    #     ],

    train_nav_db = R2RTextPathData(
        data_cfg.train_traj_files, data_cfg.img_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, in_memory=True
    )
    val_nav_db = R2RTextPathData(
        data_cfg.val_seen_traj_files, data_cfg.img_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, in_memory=True
    )
    val2_nav_db = R2RTextPathData(
        data_cfg.val_unseen_traj_files, data_cfg.img_ft_file,
        data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,
        image_prob_size=model_config.image_prob_size,
        image_feat_size=model_config.image_feat_size, 
        angle_feat_size=model_config.angle_feat_size,
        max_txt_len=opts.max_txt_len, in_memory=True
    )

    # 创建数据加载器
    # 训练数据加载器
    train_dataloaders = create_dataloaders(
        data_cfg, train_nav_db, tokenizer, True, device, opts
    )
    # 验证数据加载器（已见过的环境）
    val_dataloaders = create_dataloaders(
        data_cfg, val_nav_db, tokenizer, False, device, opts
    )
    # 验证数据加载器（未见过的环境）
    val2_dataloaders = create_dataloaders(
        data_cfg, val2_nav_db, tokenizer, False, device, opts
    )
    # 元加载器 - 管理多个任务的数据加载器
    meta_loader = MetaLoader(
        train_dataloaders,
        accum_steps=opts.gradient_accumulation_steps,
        distributed=opts.local_rank != -1,
        device=device
    )
    meta_loader = PrefetchLoader(meta_loader, device)

    # 准备优化器
    optimizer = build_optimizer(model, opts)
    task2scaler = {t: i for i, t in enumerate(train_dataloaders.keys())}

    # 设置混合精度训练
    if opts.fp16:
        grad_scaler = amp.GradScaler()

    # 开始训练
    global_step = 0
    LOGGER.info(f"***** Running training with {opts.world_size} GPUs *****")
    LOGGER.info("  Batch size = %d", opts.train_batch_size if opts.local_rank == -1 else opts.train_batch_size * opts.world_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    # 用于计算训练统计信息的变量
    task2loss = {task: RunningMeter(f'loss/{task}')
                 for task in train_dataloaders.keys()}

    n_examples = defaultdict(int)
    n_in_units = defaultdict(int)
    n_loss_units = defaultdict(int)
    grad_norm = 0

    start_time = time.time()
    # 混合精度训练的快速处理
    optimizer.zero_grad()
    optimizer.step()
    
    # 主训练循环
    for step, (name, batch) in enumerate(meta_loader):
        # 前向传播
        n_examples[name] += batch['txt_ids'].size(0)
        n_in_units[name] += batch['txt_lens'].sum().item()
        task = name.split('_')[0]
        
        # 计算损失
        if opts.fp16:
            with amp.autocast():
                loss = model(batch, task=task, compute_loss=True)
        else:
            loss = model(batch, task=task, compute_loss=True)

        n_loss_units[name] += loss.size(0)
        loss = loss.mean()  # 损失未在模型中归一化

        # 反向传播
        if args.gradient_accumulation_steps > 1: # 平均损失
            loss = loss / args.gradient_accumulation_steps

        delay_unscale = (step+1) % opts.gradient_accumulation_steps != 0
        if opts.fp16:
            grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        task2loss[name](loss.item())

        # 优化器更新和日志记录
        if (step + 1) % opts.gradient_accumulation_steps == 0:
            global_step += 1

            # 学习率调度
            lr_this_step = get_lr_sched(global_step, opts)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # 记录损失
            TB_LOGGER.log_scalar_dict({ll.name: ll.val
                                       for ll in task2loss.values()
                                       if ll.val is not None})
            TB_LOGGER.step()

            # 更新模型参数
            if opts.grad_norm != -1:
                if opts.fp16:
                    grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), opts.grad_norm
                )
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            if opts.fp16:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            pbar.update(1)

            # 定期记录训练进度
            if global_step % opts.log_steps == 0:
                # 监控训练吞吐量
                LOGGER.info(f'==============Step {global_step}===============')
                for t in train_dataloaders.keys():
                    tot_ex = n_examples[t]
                    ex_per_sec = int(tot_ex / (time.time() - start_time))
                    tot_in = n_in_units[t]
                    in_per_sec = int(tot_in / (time.time() - start_time))
                    tot_l = n_loss_units[t]
                    l_per_sec = int(tot_l / (time.time() - start_time))
                    LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                         global_step)
                    TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                         global_step)
                LOGGER.info('===============================================')

            # 定期执行验证
            if global_step % opts.valid_steps == 0:
                LOGGER.info(f'------Step {global_step}: start validation seen------')
                validate(model, val_dataloaders, setname='_seen')
                LOGGER.info(f'------Step {global_step}: start validation unseen------')
                validate(model, val2_dataloaders, setname='_unseen')
                model_saver.save(model, global_step)
        
        # 达到指定步数后结束训练
        if global_step >= opts.num_train_steps:
            break
            
    # 训练结束后进行最后一次验证并保存模型
    if global_step % opts.valid_steps != 0:
        LOGGER.info(f'------Step {global_step}: start validation seen------')
        validate(model, val_dataloaders, setname='_seen')
        LOGGER.info(f'------Step {global_step}: start validation unseen------')
        validate(model, val2_dataloaders, setname='_unseen')
        model_saver.save(model, global_step)   


def validate(model, val_dataloaders, setname=''):
    """
    在验证集上评估模型性能
    
    参数：
    - model: 模型
    - val_dataloaders: 验证数据加载器
    - setname: 数据集名称后缀（'_seen'或'_unseen'）
    """
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate val{setname} on {task} task")
        # 针对不同任务调用不同的验证函数
        if task.startswith('mlm'):
            val_log = validate_mlm(model, loader)
        elif task.startswith('mrc'):
            val_log = validate_mrc(model, loader)
        elif task.startswith('sap'):
            val_log = validate_sap(model, loader)
        else:
            raise ValueError(f'Undefined task {task}')
        # 记录验证指标
        val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
        TB_LOGGER.log_scalar_dict(
            {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()}
        )
    model.train()


@torch.no_grad()
def validate_mlm(model, val_loader):
    """
    验证MLM任务（掩码语言模型）
    
    计算预测被掩码词的准确率
    """
    LOGGER.info("start running MLM validation...")
    val_loss = 0
    n_correct = 0
    n_word = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        scores = model(batch, task='mlm', compute_loss=False)
        labels = batch['txt_labels']
        labels = labels[labels != -1]
        loss = F.cross_entropy(scores, labels, reduction='sum')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()
    # 汇总分布式训练结果
    val_loss = sum(all_gather(val_loss))
    n_correct = sum(all_gather(n_correct))
    n_word = sum(all_gather(n_word))
    tot_time = time.time()-st
    val_loss /= n_word
    acc = n_correct / n_word
    val_log = {'loss': val_loss,
               'acc': acc,
               'tok_per_s': n_word/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"acc: {acc*100:.2f}")
    return val_log

def compute_accuracy_for_soft_targets(out, labels):
    """
    计算软目标（概率分布）的准确率
    """
    outputs = out.max(dim=-1)[1]
    labels = labels.max(dim=-1)[1]  # argmax
    n_correct = (outputs == labels).sum().item()
    return n_correct

@torch.no_grad()
def validate_mrc(model, val_loader):
    """
    验证MRC任务（掩码区域分类）
    
    计算预测被掩码图像区域的准确率
    """
    LOGGER.info("start running MRC validation...")
    val_loss = 0
    n_feat = 0
    st = time.time()
    tot_score = 0
    for i, batch in enumerate(val_loader):
        view_logits, view_targets, _, _ = model(batch, task='mrc', compute_loss=False)
        view_logprobs = F.log_softmax(view_logits, dim=-1)
        loss = F.kl_div(view_logprobs, view_targets, reduction='sum')
        tot_score += compute_accuracy_for_soft_targets(view_logits, view_targets)
        val_loss += loss.item()
        n_feat += batch['vp_view_mrc_masks'].sum().item()
    # 汇总分布式训练结果
    val_loss = sum(all_gather(val_loss))
    tot_score = sum(all_gather(tot_score))
    n_feat = sum(all_gather(n_feat))
    tot_time = time.time()-st
    val_loss /= n_feat
    val_acc = tot_score / n_feat
    val_log = {'loss': val_loss,
               'acc': val_acc,
               'feat_per_s': n_feat/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc*100:.2f}")
    return val_log
    
@torch.no_grad()
def validate_sap(model, val_loader):
    """
    验证SAP任务（行为预测）
    
    计算预测下一步导航行为的准确率
    """
    LOGGER.info("start running SAP validation...")
    val_gloss, val_lloss, val_floss = 0, 0, 0
    n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
    n_data = 0
    st = time.time()
    for i, batch in enumerate(val_loader):
        # 获取全局、局部和融合的预测结果
        global_logits, local_logits, fused_logits, global_act_labels, local_act_labels = \
            model(batch, task='sap', compute_loss=False)
        # 计算损失
        val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
        val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
        val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()
        # 计算正确预测数
        n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
        n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
        n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
        n_data += len(global_act_labels)

    # 汇总分布式训练结果
    n_data = sum(all_gather(n_data))
    val_gloss = sum(all_gather(val_gloss)) / n_data
    val_lloss = sum(all_gather(val_lloss)) / n_data
    val_floss = sum(all_gather(val_floss)) / n_data
    gacc = sum(all_gather(n_gcorrect)) / n_data
    lacc = sum(all_gather(n_lcorrect)) / n_data
    facc = sum(all_gather(n_fcorrect)) / n_data
    
    tot_time = time.time()-st
    val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
               'gacc': gacc, 'lacc': lacc, 'facc': facc,
               'tok_per_s': n_data/tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
    return val_log

def build_args():
    """
    构建命令行参数
    """
    parser = load_parser()
    #从“duet/pretrain_src/config/r2r_pretrain.json” 中读取配置
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Output directory ({}) already exists and is not empty.".format(
                opts.output_dir
            )
        )

    return opts

if __name__ == '__main__':
    args = build_args()
    main(args)