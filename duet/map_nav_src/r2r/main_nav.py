import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
import sys
sys.path.append("/projects/VLN-Tutorial/duet/map_nav_src")
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB
from r2r.data_utils import construct_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args

from models.vlnbert_init import get_tokenizer
from r2r.agent import GMapNavAgent


def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = R2RNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, 
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
        )
    else:
        aug_env = None

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'], 
        tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', 
    )

    # val_env_names = ['val_train_seen']
    val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
    if args.dataset == 'r4r' and (not args.test):
        val_env_names[-1] == 'val_unseen_sampled'
    
    if args.submit and args.dataset != 'r4r':
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    """
    训练函数：实现模型训练和验证的主循环
    """
    # 初始化：日志、模型、记录文件等
    default_gpu = is_default_gpu(args)  # 确定是否为主GPU进程
    if default_gpu:
        # 设置日志记录
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    # 创建导航智能体
    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank)

    # 如果有指定，从检查点恢复训练
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
       
    # 可选的初始评估（训练前验证）
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
        # return

    # 记录训练开始时间
    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    # 记录最佳验证结果
    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
    if args.dataset == 'r4r':
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state":""}}
    
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # 训练log_every轮次
        if aug_env is None:
            # 使用原始训练环境
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # 训练interval次迭代
        else:
            # 使用原始数据和增强数据交替训练
            for jdx in range(interval // 2):
                # 使用原始数据训练
                listner.env = train_env
                listner.train(1, feedback=args.feedback)

                # 使用增强数据训练
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, interval // 2, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # 计算各种损失值，并记录到tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )

        # ===== 验证阶段 =====
        # 在每个验证环境上评估模型
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env
            # 在测试条件下评估
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            
            # 合并分布式结果（多GPU情况）
            preds = merge_dist_results(all_gather(preds))
            
            if default_gpu:
                # 计算评估指标
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # 根据SPL指标选择最佳模型
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        # 更新最佳记录
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        # 保存最佳模型
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
                


def main():
    """
    主函数：程序入口点
    初始化环境并根据参数执行训练或测试流程
    """
    # 解析命令行参数
    args = parse_args()

    # 分布式训练设置（多GPU）
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    # 设置随机种子确保可重复性
    set_random_seed(args.seed + rank)
    
    # 构建数据集：训练环境、验证环境和可选的增强数据环境
    train_env, val_envs, aug_env = build_dataset(args, rank=rank, is_test=args.test)

    # 根据参数选择训练或测试模式
    if not args.test:
        # 训练模式
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        # 测试模式
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    main()
