import argparse
import os
import numpy as np
from datetime import datetime
from CMF_KM import CMF
from GMF_AM import GMF_BOOSTING
from DatasetLoading import get_k_fold_split_sequences

def parse_args():
    parser = argparse.ArgumentParser(description="ABKT with K-Fold Cross Validation")
    
    parser.add_argument('--dataset', default='ASSISTment2009',
                        help='Choose dataset. Default is "ASSISTment2009", choose from ["ASSISTment2009","AICFE"]. ')
    
    parser.add_argument('--type', default='all_data',
                        help='The subset of the dataset.')
    
    parser.add_argument('--KM_k', type=int, default=5,
                        help='The dimensionality of knowledge module, namely k_K, Default is 5')
    
    parser.add_argument('--KM_guess', type=float, default=0.25,
                        help='The surmise of knowledge module. Default is 0.25.')
    
    parser.add_argument('--AM_k', type=int, default=64,
                        help='The dimensionality of ability model, namely k_A,. Default is 64.')
    
    parser.add_argument('--AM_lambda', type=float, default=0.1,
                        help='The control factor of regularization term in ability model. Default is 0.1.')
    
    parser.add_argument('--AM_layer', type=int, default=1,
                        help='The depth of feature aggregation in the ability module. Default is 1.')
    
    parser.add_argument('--pretrain_clip', type=float, default=0.4,
                        help='The clip range, namly _mu. Default is 0.4.')
    
    parser.add_argument('--joint_model', default="add",
                        help='The type of the joint model. Default is "add". Choose from ["add","mul"]. ')
    
    parser.add_argument('--device', default="cuda:0",
                        help='The working device of pytorch. Default is "cuda:0"')
    
    parser.add_argument('--use_kfold', type=bool, default=True,
                        help='Whether to use k-fold cross validation. Default is True.')
    
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross validation. Default is 5.')
    
    return parser.parse_args()

def log_results(results, args, model_paths=None):
    """保存k折交叉验证结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"./Results/KFold_{args.dataset}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存详细结果
    with open(f"{result_dir}/kfold_results.txt", "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Type: {args.type}\n")
        f.write(f"K-Fold: {args.k_folds}\n")
        f.write(f"KM_k: {args.KM_k}, KM_guess: {args.KM_guess}\n")
        f.write(f"AM_k: {args.AM_k}, AM_lambda: {args.AM_lambda}\n")
        f.write(f"Joint Model: {args.joint_model}\n")
        f.write("="*50 + "\n\n")
        
        # 每折结果
        for i, result in enumerate(results):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  CMF - ACC: {result['CMF_ACC']:.4f}, AUC: {result['CMF_AUC']:.4f}\n")
            f.write(f"  GMF - ACC: {result['GMF_ACC']:.4f}, AUC: {result['GMF_AUC']:.4f}\n")
            f.write("\n")
        
        # 平均结果
        cmf_accs = [r['CMF_ACC'] for r in results]
        cmf_aucs = [r['CMF_AUC'] for r in results]
        gmf_accs = [r['GMF_ACC'] for r in results]
        gmf_aucs = [r['GMF_AUC'] for r in results]
        
        f.write("="*50 + "\n")
        f.write("Average Results:\n")
        f.write(f"CMF - ACC: {np.mean(cmf_accs):.4f} ± {np.std(cmf_accs):.4f}\n")
        f.write(f"CMF - AUC: {np.mean(cmf_aucs):.4f} ± {np.std(cmf_aucs):.4f}\n")
        f.write(f"GMF - ACC: {np.mean(gmf_accs):.4f} ± {np.std(gmf_accs):.4f}\n")
        f.write(f"GMF - AUC: {np.mean(gmf_aucs):.4f} ± {np.std(gmf_aucs):.4f}\n")
        
        # 如果提供了模型路径，也保存进去
        if model_paths:
            f.write("\n" + "="*50 + "\n")
            f.write("Saved Model Paths:\n")
            for i, paths in enumerate(model_paths):
                f.write(f"Fold {i+1}:\n")
                f.write(f"  CMF: {paths['CMF']}\n")
                f.write(f"  GMF: {paths['GMF']}\n")
    
    print(f"Results saved to {result_dir}")
    return result_dir

def main():
    args = parse_args()
    
    if args.use_kfold:
        print(f"Starting {args.k_folds}-fold cross validation...")
        
        # 获取k折数据
        folds = get_k_fold_split_sequences(
            dataset=args.dataset,
            type=args.type,
            min_length=10,
            k=args.k_folds
        )
        
        all_results = []
        all_model_paths = []
        
        for fold_idx, (train_sequences, test_triplet, Q_matrix) in enumerate(folds):
            print(f"\n{'='*60}")
            print(f"Fold {fold_idx + 1}/{args.k_folds}")
            print(f"{'='*60}")
            
            # 训练知识模块
            print("Training knowledge module...")
            cmf = CMF(
                dataset=args.dataset,
                type=args.type,
                k_hidden_size=args.KM_k,
                guess=args.KM_guess,
                device=args.device,
                use_kfold=True,
                kfold_data=(train_sequences, test_triplet, Q_matrix),
                fold_idx=fold_idx
            )
            cmf.train()
            
            # 训练能力模块
            print("Training ability module...")
            gmf = GMF_BOOSTING(
                dataset=args.dataset,
                type=args.type,
                CMF_k=args.KM_k,
                CMF_guess=args.KM_guess,
                embedding_k=args.AM_k,
                m_lambda=args.AM_lambda,
                GMF_layer=args.AM_layer,
                pretrain_clip=args.pretrain_clip,
                combine=args.joint_model,
                device=args.device,
                use_kfold=True,
                kfold_data=(train_sequences, test_triplet, Q_matrix),
                fold_idx=fold_idx,
                cmf_model=cmf.K_CMF  # 传递训练好的CMF模型
            )
            gmf.train()
            
            # 记录模型路径
            fold_model_paths = {
                'CMF': f'./Models/{args.dataset}-{args.type}/CMF-k-{args.KM_k}-{args.KM_guess}-fold{fold_idx}-earlystop',
                'GMF': f'./Models/{args.dataset}-{args.type}/GMF-boosting-{args.joint_model}-{args.AM_k}-fold{fold_idx}-earlystop'
            }
            all_model_paths.append(fold_model_paths)
            
            # 记录结果
            fold_result = {
                'fold': fold_idx + 1,
                'CMF_ACC': cmf.bestACC,
                'CMF_AUC': cmf.bestAUC,
                'GMF_ACC': gmf.bestACC,
                'GMF_AUC': gmf.bestAUC
            }
            all_results.append(fold_result)
            
            print(f"Fold {fold_idx + 1} Results:")
            print(f"  CMF - ACC: {cmf.bestACC:.4f}, AUC: {cmf.bestAUC:.4f}")
            print(f"  GMF - ACC: {gmf.bestACC:.4f}, AUC: {gmf.bestAUC:.4f}")
        
        # 保存结果和模型路径
        result_dir = log_results(all_results, args, all_model_paths)
        
        print(f"\n{'='*60}")
        print("K-Fold Cross Validation Summary")
        print(f"{'='*60}")
        
        cmf_accs = [r['CMF_ACC'] for r in all_results]
        cmf_aucs = [r['CMF_AUC'] for r in all_results]
        gmf_accs = [r['GMF_ACC'] for r in all_results]
        gmf_aucs = [r['GMF_AUC'] for r in all_results]
        
        print(f"CMF - ACC: {np.mean(cmf_accs):.4f} ± {np.std(cmf_accs):.4f}")
        print(f"CMF - AUC: {np.mean(cmf_aucs):.4f} ± {np.std(cmf_aucs):.4f}")
        print(f"GMF - ACC: {np.mean(gmf_accs):.4f} ± {np.std(gmf_accs):.4f}")
        print(f"GMF - AUC: {np.mean(gmf_aucs):.4f} ± {np.std(gmf_aucs):.4f}")
        
    else:
        # 原有的单次训练逻辑
        print("Single training mode...")
        
        # 检查预训练模型
        KM_path = './Models/' + str(args.dataset) + '-' + str(args.type) + '/CMF-k-' + str(args.KM_k) + '-' + str(args.KM_guess) + '-earlystop'
        if os.access(KM_path, os.F_OK):
            print("Pre-trained knowledge model exists...")
        else:
            print("Training knowledge model...")
            cmf = CMF(
                dataset=args.dataset,
                type=args.type,
                k_hidden_size=args.KM_k,
                guess=args.KM_guess,
                device=args.device,
            )
            cmf.train()
            cmf.log_result()
        
        print("Training boosted ability model...")
        gmf = GMF_BOOSTING(
            dataset=args.dataset,
            type=args.type,
            CMF_k=args.KM_k,
            CMF_guess=args.KM_guess,
            embedding_k=args.AM_k,
            m_lambda=args.AM_lambda,
            GMF_layer=args.AM_layer,
            pretrain_clip=args.pretrain_clip,
            combine=args.joint_model,
            device=args.device,
        )
        gmf.train()
        gmf.log_result()

if __name__ == "__main__":
    main()
