from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from eval_metrics import eval_sysu, eval_regdb
from tacnet import TACNet
from utils import *
from loss import DCALLoss, GLPRLoss, CCMLoss
from ChannelAug import ChannelAdapGray, ChannelRandomErasing
from sklearn.mixture import GaussianMixture
import datetime
import tarfile

def gen_code_archive(out_dir, file='code.tar.gz'):
    archive = os.path.join(out_dir, file)
    print(f"code save in {archive}")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    with tarfile.open(archive, mode='w:gz') as tar:
        tar.add('.', filter=is_source_file)
    return archive

def is_source_file(x):
    if x.isdir() or x.name.endswith(('.py', '.sh', '.yml', '.json', '.txt', '.md')) \
            and '.mim' not in x.name and 'jobs/' not in x.name:
        return x
    else:
        return None


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training (TACNet)')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume-net1', default='', type=str,
                    help='resume net1 from checkpoint')
parser.add_argument('--model_path', default='./save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=1, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='robust', type=str,
                    metavar='m', help='method type: base or agw or robust')

parser.add_argument('--loss1', default='dcal', type=str, help='loss type: dcal (new) or id')
parser.add_argument('--loss2', default='robust_tri', type=str,
                    metavar='m', help='loss type: wrt or adp or robust_tri')
parser.add_argument('--margin', default=0.2, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--savename', default='TACNet_sysu_batch_6_384*192', type=str,
                    help='name of the saved model')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--augc', default=1, type=int,
                    metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default=0.5, type=float,
                    metavar='ra', help='use random erasing or not and the probability')
parser.add_argument('--kl', default=0, type=float,
                    metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1, type=int,
                    metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1, type=int,
                    metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default=1, type=int,
                    metavar='square', help='gamma for the hard mining')
parser.add_argument('--noise-mode', default='sym', type=str, help='sym')
parser.add_argument('--noise-rate', default=0.90, type=float,
                    metavar='nr', help='noise_rate')
parser.add_argument('--data-path', default='./dataset/', type=str, help='path to dataset')

parser.add_argument('--p-threshold', default=0.6, type=float, help='clean probability threshold (DCAL)')
parser.add_argument('--warm-epoch', default=1, type=int, help='epochs for net warming up')
parser.add_argument('--UseCL', default=0, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--scale', default=2, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')

parser.add_argument('--glpr_w', default=1.0, type=float, help='weight of GLPR loss')
parser.add_argument('--ccm_w', default=0.2, type=float, help='weight of CCM loss')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
data_path = args.data_path

if dataset == 'sysu':
    test_mode = [1, 2]
    log_path = args.log_path + 'sysu_log_tacnet/'

elif dataset == 'regdb':
    test_mode = [2, 1]
    log_path = args.log_path + 'regdb_log_tacnet/'

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.datetime.today().strftime(fmt)

timestamp = time_str()
checkpoint_path = args.model_path

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(log_path):
    os.makedirs(log_path)

sys.stdout = Logger(log_path + args.savename+timestamp + '.txt')
gen_code_archive(out_dir=os.path.join('results/', ), file=f'{ args.savename+timestamp}.tar.gz')

with open('./semi_main.py', encoding='utf-8') as file:
    content = file.read()
    print(content.rstrip())
file.close()
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize]

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize])

if args.rande > 0:
    transform_train_list = transform_train_list + [ChannelRandomErasing(probability=args.rande)]

if args.augc == 1:
    transform_train_list = transform_train_list + [ChannelAdapGray(probability=0.5)]

transform_train = transforms.Compose(transform_train_list)

end = time.time()
if dataset == 'sysu':
    evaltrainset = SYSUData(data_path, transform=transform_test, noise_rate=args.noise_rate,
                            noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                            mode='evaltrain')
    color_pos, thermal_pos = GenIdx(evaltrainset.train_color_label, evaltrainset.train_thermal_label)
    create_pesudoset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='create_pesudo',
                                noise_file='%s/%.2f_%s' % (data_path, args.noise_rate, args.noise_mode))
    pre_trainset = SYSUData(data_path,  transform=transform_train, noise_rate=args.noise_rate,
                          noise_file='%s/%.2f_%s' % (data_path, args.noise_rate, args.noise_mode),
                          mode='pre_train')
    warmupset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='warmup',
                         noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode))
    trainset = SYSUData(data_path, transform=transform_train, noise_rate=args.noise_rate, mode='train',
                        noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode))

    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    evaltrainset = RegDBData(data_path,
                             trial=args.trial,
                             transform=transform_test,
                             noise_rate=args.noise_rate,
                             noise_file='%s/%.2f_%s' % (args.data_path, args.noise_rate, args.noise_mode),
                             mode='evaltrain')
    color_pos, thermal_pos = GenIdx(evaltrainset.train_color_label, evaltrainset.train_thermal_label)
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(evaltrainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(evaltrainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(evaltrainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


print('==> Building TACNet model..')
net1 = TACNet(n_class, no_local='on', gm_pool='on', arch=args.arch)
net1.to(device)
cudnn.benchmark = True


criterion_id = nn.CrossEntropyLoss()
criterion_CE = nn.CrossEntropyLoss(reduction='none')

criterion_dcal = DCALLoss(cls_thresh=args.p_threshold).to(device)
criterion_glpr = GLPRLoss().to(device)
criterion_ccm = CCMLoss().to(device)

loader_batch = args.batch_size * args.num_pos
criterion2 = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

if args.loss2 == 'wrt':
    criterion_tri = TripletLoss_WRT()
    criterion_tri.to(device)
elif args.loss2 == 'adp':
    criterion_tri = TripletLoss_ADP(alpha=args.alpha, gamma=args.gamma, square=args.square)
    criterion_tri.to(device)
elif args.loss2 == 'robust_tri':
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = RobustTripletLoss_final(batch_size=loader_batch, margin=args.margin)
criterion_l1 = nn.L1Loss()
MMDLoss = MMD_loss().to(device)


if args.optim == 'sgd':
    ignored_params1 = list(map(id, net1.bottleneck.parameters())) \
                      + list(map(id, net1.classifier.parameters()))
    base_params1 = filter(lambda p: id(p) not in ignored_params1, net1.parameters())
    optimizer1 = optim.SGD([
        {'params': base_params1, 'lr': 0.1 * args.lr},
        {'params': net1.bottleneck.parameters(), 'lr': args.lr},
        {'params': net1.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


if len(args.resume_net1) > 0:
    model_path1 = checkpoint_path + args.resume_net1
    if os.path.isfile(model_path1):
        print('==> loading checkpoint {}'.format(args.resume_net1))
        checkpoint1 = torch.load(model_path1)
        net1.load_state_dict(checkpoint1['net'])
        start_epoch = checkpoint1['epoch'] + 1
        optimizer1.load_state_dict(checkpoint1['optimizer'])
    else:
        print('==> no checkpoint found at {} '.format(args.resume_net1))

def adjust_learning_rate(optimizer, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    return lr

def pre_train(epoch, net1,net2, optimizer, dataloader):
    current_lr = adjust_learning_rate(optimizer, epoch)
    net1.train()
    net2.train()
    num_iter = (len(dataloader.dataset.cIndex) // dataloader.batch_size) + 1
    for epoch_pre in range(epoch):
        for batch_idx, (input10, input11, input2, label1, label2) in enumerate(dataloader):
            n=input10.size(0)
            labels1 = torch.cat((label1, label1), 0)
            labels2 = torch.cat((label2, label2), 0)
            input10 = input10.cuda()
            input11 = input11.cuda()
            input2 = input2.cuda()
            labels1 = labels1.cuda().long()
            labels2 = labels2.cuda().long()
            

            _, out0 = net1(input10, input11)
            _, out1 = net2(input2, input2)

            loss_id0 = criterion_id(out0, labels1)
            loss_id1 = criterion_id(out1, labels2)

            optimizer.zero_grad()
            loss_id0.backward()
            loss_id1.backward()
            optimizer.step()

            predict0 = out0.argmax(dim=1)
            predict1 = out1.argmax(dim=1)
            res_rgb = labels1 - predict0
            res_rgb = np.array(res_rgb.cpu())
            res_rgb = np.sum(res_rgb == 0)
            acc_rgb=res_rgb/(2*n)

            res_ir = labels2 - predict1
            res_ir = np.array(res_ir.cpu())
            res_ir = np.sum(res_ir == 0)
            acc_ir = res_ir / (2*n)

            if batch_idx % 50 == 0:
                print('%s:%.2f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f\t Current-lr: %.4f \t RGB-acc: %.4f \t IR-acc: %.4f'
                      % (args.dataset, args.noise_rate, args.noise_mode, epoch_pre, 10, batch_idx + 1,
                         num_iter, loss_id1.item(), current_lr,acc_rgb,acc_ir))
        state = {
            'net': net1.state_dict(),
            'epoch': epoch_pre,
        }
    return net1

def warmup(epoch, net, optimizer, dataloader):
    current_lr = adjust_learning_rate(optimizer, epoch)
    net.train()
    num_iter = (len(dataloader.dataset.cIndex) // dataloader.batch_size) + 1
    for batch_idx, (input10, input11, input2, label1, label2) in enumerate(dataloader):
        labels = torch.cat((label1, label1, label2), 0)
        input1 = torch.cat((input10, input11,), 0)
        input1 = input1.cuda()
        input2 = input2.cuda()
        labels = labels.cuda()
        
        # 适配 TACNet
        _, out0 = net(input1, input2)
        loss_id = criterion_id(out0, labels)

        optimizer.zero_grad()
        loss_id.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('%s:%.1f-%s | Warmup Epoch [%3d/80] Iter[%3d/%3d]\t CE-loss: %.4f\t LR: %.4f'
                  % (args.dataset, args.noise_rate, args.noise_mode, epoch, batch_idx + 1,
                     num_iter, loss_id.item(), current_lr))

def eval_train(net, dataloader, type):
    losses_V_aug1 = -1. * torch.ones(len(evaltrainset.train_color_label))
    losses_V_aug2 = -1. * torch.ones(len(evaltrainset.train_color_label))
    losses_I = -1. * torch.ones(len(evaltrainset.train_thermal_label))

    net.train()
    with torch.no_grad():
        for batch_idx, (input10, input11, input2, label1, label2, index_V, index_I) in enumerate(dataloader):
            input1 = torch.cat((input10, input11,), 0)
            input1 = input1.cuda()
            input2 = input2.cuda()
            label1 = label1.cuda()
            label2 = label2.cuda()

            index_V = np.concatenate((index_V, index_V), 0)
            labels = torch.cat((label1, label1, label2), 0)
            _, out0 = net(input1, input2)
            loss = criterion_CE(out0, labels)
            loss1 = loss[0:input2.size(0)*2]
            loss2 = loss[input2.size(0)*2:input2.size(0)*3]

            for n1 in range(input2.size(0)):
                losses_V_aug1[index_V[n1]] = loss1[n1]
                losses_V_aug2[index_V[n1 + input2.size(0)]] = loss1[n1 + input2.size(0)]
            for n2 in range(input2.size(0)):
                losses_I[index_I[n2]] = loss2[n2]

    losses_V_aug1_slt = (losses_V_aug1 - losses_V_aug1.min()) / (losses_V_aug1.max() - losses_V_aug1.min())
    losses_V_aug2_slt = (losses_V_aug2 - losses_V_aug2.min()) / (losses_V_aug2.max() - losses_V_aug2.min())
    losses_I_slt = (losses_I - losses_I.min()) / (losses_I.max() - losses_I.min())
    losses_V_slt = torch.cat((losses_V_aug1_slt, losses_V_aug2_slt), 0)

    input_loss_V = losses_V_slt.reshape(-1, 1)
    input_loss_I = losses_I_slt.reshape(-1, 1)

    gmm_V = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm_V.fit(input_loss_V)
    prob_V = gmm_V.predict_proba(input_loss_V)
    prob_V = prob_V[:, gmm_V.means_.argmin()]

    gmm_I = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    gmm_I.fit(input_loss_I)
    prob_I = gmm_I.predict_proba(input_loss_I)
    prob_I = prob_I[:, gmm_I.means_.argmin()]

    return prob_V, prob_I

def create_pesudo(net, dataloader,noise,dataset):
    net.train()
    with torch.no_grad():
        acc_irs=0
        acc_rgbs=0
        for batch_idx, (input1, input2, label1, label2, cindex, tindex) in enumerate(dataloader):
            n=int(input1.size(0))
            labels = torch.cat((label1, label2), 0)
            input1 = input1.cuda()
            input2 = input2.cuda()
            labels = labels.cuda()
            _, out0 = net1(input1, input2)
            predict = out0.argmax(dim=1)
            res = labels - predict
            res = np.array(res.cpu())
            res_rgb = np.sum(res[0:n] == 0)
            res_ir = np.sum(res[n:] == 0)
            acc_rgb = res_rgb / n
            acc_ir = res_ir / n
            acc_irs+=acc_ir
            acc_rgbs+=acc_rgb

            warmupset.train_color_label[cindex] = predict[0:n].cpu()
            warmupset.train_thermal_label[tindex] = predict[n:].cpu()
            
        rgb_predictlabels=warmupset.train_color_label
        rgb_turelabels=warmupset.true_train_color_label
        res = rgb_predictlabels - rgb_turelabels
        res = np.array(res)
        res_rgb = np.sum(res == 0)
        acc_rgb = res_rgb / len(rgb_predictlabels)
        print("Pseudo Label RGB Acc:",acc_rgb)
        
        ir_predictlabels = warmupset.train_thermal_label
        ir_turelabels = warmupset.true_train_thermal_label
        res = ir_predictlabels - ir_turelabels
        res = np.array(res)
        ir_rgb = np.sum(res == 0)
        acc_ir = ir_rgb / len(ir_predictlabels)
        print("Pseudo Label IR Acc:", acc_ir)
        
        trainset.train_color_label=rgb_predictlabels
        trainset.train_thermal_label=ir_predictlabels
        evaltrainset.train_color_label = rgb_predictlabels
        evaltrainset.train_thermal_label= ir_predictlabels


def train(epoch, net, optimizer, trainloader):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    glpr_loss_meter = AverageMeter()  
    ccm_loss_meter = AverageMeter()   
    data_time = AverageMeter()
    batch_time = AverageMeter()
    cnt_sum = 0
    id_correct = 0
    tri_correct = 0
    total = 0

    net.train()
    end = time.time()
    for batch_idx, (input10, input11, input2, label1, label2, true_label1, true_labels2, probV_1, probV_2,
                    probI) in enumerate(trainloader):
        labels = torch.cat((label1, label1, label2), 0)
        input1 = torch.cat((input10, input11,), 0)
        true_labels = torch.cat((true_label1, true_label1, true_labels2), 0)
        probV_1 = probV_1.cuda()
        probV_2 = probV_2.cuda()
        probI = probI.cuda()
        prob = torch.cat((probV_1, probV_2, probI), 0)
        input1 = input1.cuda()
        input2 = input2.cuda()
        labels = labels.cuda()
        prob = prob.cuda()
        data_time.update(time.time() - end)
        

        pool_feat, refined_feat, logits, ccm_l, pseudo_pids, valid_mask, adaptive_w = net(input1, input2, pids=labels)

        
        loss_total = 0
        

        loss_tri1, batch_acc, cnt = criterion_tri(refined_feat, logits, labels, true_labels, prob, threshold=args.p_threshold)
        loss_tri, _ = criterion2(refined_feat, labels)
        

        loss_glpr = criterion_glpr(refined_feat, pool_feat) * args.glpr_w
        

        loss_ccm = ccm_l * args.ccm_w
        

        loss_total = loss_id + loss_tri1 + loss_glpr + loss_ccm


        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()


        net.glpr.refresh_prototypes()
        net.dcal.refresh_weights()


        train_loss.update(loss_total.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        glpr_loss_meter.update(loss_glpr.item(), 2 * input1.size(0))
        ccm_loss_meter.update(loss_ccm.item(), 2 * input1.size(0))
        
        total += labels.size(0)
        cnt_sum += int(cnt)
        tri_correct += batch_acc
        _, predicted = logits.max(1)
        id_correct += predicted.eq(labels).sum().item()

        batch_time.update(time.time() - end)
        end = time.time()
        

        if batch_idx % 50 == 0:
            print('TACNet Epoch: [{}][{}/{}] '
                  'Time: {batch_time.avg:.3f} '
                  'LR:{:.3f} '
                  'TotalLoss: {train_loss.avg:.4f} '
                  'DCAL: {id_loss.avg:.4f} '
                  'Triplet: {tri_loss.avg:.4f} '
                  'GLPR: {glpr.avg:.4f} '
                  'CCM: {ccm.avg:.4f} '
                  'ID-Acc: {:.2f} '
                  'Tri-Acc: {:.2f} '.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * id_correct / total, 100. * tri_correct / cnt_sum,
                batch_time=batch_time, train_loss=train_loss, 
                id_loss=id_loss, tri_loss=tri_loss,
                glpr=glpr_loss_meter, ccm=ccm_loss_meter))
    return 1. / (1. + train_loss.avg)

# 测试函数（无需修改，TACNet 测试输出兼容原结构）
def test(net1):
    net1.eval()
    ptr = 0
    gall_feat_att = np.zeros((ngall, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att = net1(input, input, test_mode[0])
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    ptr = 0
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = input.cuda()
            _, feat_att = net1(input, input, test_mode[1])
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num

    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    if dataset == 'regdb':
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, query_cam, gall_cam)

    return cmc_att, mAP_att, mINP_att



if args.noise_rate>=0.0001:
    print('==> Preparing Data Loader...')
    loader_batch = args.batch_size * args.num_pos
    model_path=''
    print('==> Start Pre-training...')

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        net1.load_state_dict(checkpoint['net'])
        print('==> Loaded pre-trained model')
    else:
        pre_sampler = AllSampler(args.dataset, pre_trainset.train_color_label, pre_trainset.train_thermal_label)
        pre_trainset.cIndex = pre_sampler.index1
        pre_trainset.tIndex = pre_sampler.index2
        pre_trainloader = data.DataLoader(pre_trainset, batch_size=24, sampler=pre_sampler, 
                                          num_workers=args.workers, drop_last=False)
        net1 = pre_train(31, net1,net1, optimizer1, pre_trainloader)
    print('==> Pre-training finished!')

    print('==> Generating pseudo labels...')
    create_pesudo_label_sampler = AllSampler(args.dataset, create_pesudoset.train_color_label,
                                             create_pesudoset.train_thermal_label)
    create_pesudoset.cIndex = create_pesudo_label_sampler.index1
    create_pesudoset.tIndex = create_pesudo_label_sampler.index2
    create_pesudolabel_loader = data.DataLoader(create_pesudoset,
                                                batch_size=loader_batch,
                                                sampler=create_pesudo_label_sampler,
                                                num_workers=args.workers,
                                                drop_last=False)
    create_pesudo(net1, create_pesudolabel_loader,'%.2f' % (args.noise_rate),dataset)
    print('==> Pseudo labels generated!')
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

for epoch in range(start_epoch, 81):
    print('==> Preparing Data Loader...')
    loader_batch = args.batch_size * args.num_pos

    if epoch < args.warm_epoch:
        warmup_path1 = ''
        if os.path.isfile(warmup_path1):
            checkpoint1 = torch.load(warmup_path1)
            net1.load_state_dict(checkpoint1['net'])
        else:
            warm_sampler = AllSampler(args.dataset, warmupset.train_color_label, warmupset.train_thermal_label)
            warmupset.cIndex = warm_sampler.index1
            warmupset.tIndex = warm_sampler.index2
            warmup_trainloader = data.DataLoader(warmupset, batch_size=loader_batch, sampler=warm_sampler, 
                                                 num_workers=args.workers, drop_last=True)
            print('Warming up TACNet...')
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print('\n')

    else:
        if epoch%1==0 and args.noise_rate>=0.0001:
            create_pesudo(net1, create_pesudolabel_loader, '%.2f' % (args.noise_rate), dataset)
            color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        
        eval_sampler = AllSampler(args.dataset, evaltrainset.train_color_label, evaltrainset.train_thermal_label)
        evaltrainset.cIndex = eval_sampler.index1
        evaltrainset.tIndex = eval_sampler.index2
        eval_loader = data.DataLoader(evaltrainset, batch_size=loader_batch, sampler=eval_sampler,
                                      num_workers=args.workers, drop_last=True)

        prob_A_V, prob_A_I = eval_train(net1, eval_loader, 'A')
        print('Training TACNet...')
        trainset.probV_1, trainset.probV_2, trainset.probI = prob_A_V[0:int(len(prob_A_V) / 2)], prob_A_V[int(len(prob_A_V) / 2):], prob_A_I

        train_sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label,
                                        color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)
        trainset.cIndex = train_sampler.index1
        trainset.tIndex = train_sampler.index2
        trainloader = data.DataLoader(dataset=trainset, batch_size=loader_batch,
                                      num_workers=args.workers, sampler=train_sampler, drop_last=True)

        train(epoch, net1, optimizer1, trainloader)

    if epoch != 0 or epoch == 99:
        print('Testing Epoch: {}'.format(epoch))
        cmc_att, mAP_att, mINP_att = test(net1)
        state = {
            'net': net1.state_dict(),
            'cmc': cmc_att,
            'mAP': mAP_att,
            'mINP': mINP_att,
            'epoch': epoch,
            'optimizer': optimizer1.state_dict()
        }

        if epoch >= args.warm_epoch or epoch == 99:
            if args.dataset == 'sysu':
                torch.save(state, checkpoint_path + args.savename + timestamp + '_TACNet.t')
            else:
                torch.save(state, checkpoint_path + args.savename + '_trial{}_TACNet.t'.format(args.trial))

        print('Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'
              .format(cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))

        if cmc_att[0] > best_acc:
            best_acc = cmc_att[0]
            torch.save(state, checkpoint_path  + args.savename+timestamp+'_TACNet_best.t')
            print('[BEST] Rank-1: {:.2%} | mAP: {:.2%}| mINP: {:.2%}'
                  .format(cmc_att[0], mAP_att, mINP_att))