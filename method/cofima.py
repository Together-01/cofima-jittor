import logging
import os
import numpy as np

import jittor as jt
import jittor.nn as nn
from jittor.dataset import DataLoader

from utils.inc_net import ViTIncrementalModel, load_pretrained_backbone
from utils.data_manager import DataManager


class CoFiMA(object):
    def __init__(self, args):
        self.dataset = args["dataset"]
        self.shuffle = args["shuffle"]
        self.seed = args["seed"]
        self.init_cls = args["init_cls"]
        self.increment = args["increment"]

        self.model_name = args["model_name"]
        self.pretrained_path = args.get("pretrained_path", "")

        self.batch_size = args["batch_size"]
        self.fisher_batch_size = args.get("fisher_batch_size", max(1, self.batch_size // 2))
        self.num_workers = args["num_workers"]
        self.epochs = args["epochs"]
        self.lr = args["lr"]
        self.weight_decay = args["weight_decay"]
        self.milestones = args.get("milestones", [60, 100, 140])
        self.lrate_decay = args.get("lrate_decay", 0.1)
        self.bcb_lrscale = args.get("bcb_lrscale", 1.0 / 100)
        self.fix_bcb = self.bcb_lrscale == 0
        self.wt_alpha = args.get("wt_alpha", 0.5)
        self.init_w = args.get("init_w", -1)
        self.fisher_weighting = args.get("fisher_weighting", args.get("fisher-weighting", True))
        self.fisher_max = args.get("fisher_max", 1e-4)
        self.manual_release = args.get("manual_release", False)

        self.ca_epochs = args.get("ca_epochs", 5)
        self.ca_samples_per_class = args.get("ca_samples_per_class", 256)
        ca_with_logit_norm = args.get("ca_with_logit_norm", 0)
        self.logit_norm = ca_with_logit_norm if ca_with_logit_norm and ca_with_logit_norm > 0 else None

        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = ViTIncrementalModel(self.model_name)
        load_pretrained_backbone(self._network.convnet, self.pretrained_path)
        self.prev_nets = []
        self.fisher_mat = []
        self.top1_curve = []
        self.task_sizes = []
        self._class_means = None
        self._class_covs = None

    def run(self):
        data_manager = DataManager(self.dataset, self.shuffle, self.seed, self.init_cls, self.increment)
        logging.info("Data manager ready, total tasks: {}".format(data_manager.nb_tasks))

        for _ in range(data_manager.nb_tasks):
            top1_acc = self.inc_train(data_manager)
            self.top1_curve.append(top1_acc)
            self.after_task()

        if len(self.top1_curve) > 0:
            avg_top1 = sum(self.top1_curve) / len(self.top1_curve)
            logging.info("")
            logging.info("Top1 curve: {}".format(self.top1_curve))
            logging.info("Inc-Acc: {:.3f}".format(avg_top1))
            logging.info("Last-Acc: {:.3f}".format(self.top1_curve[-1]))

    def inc_train(self, data_manager):
        self._cur_task += 1
        start = sum(data_manager._increments[: self._cur_task])
        end = start + data_manager.get_task_size(self._cur_task)
        task_classes = list(range(start, end))
        seen_classes = list(range(end))
        logging.info("")
        logging.info("Learning on {}-{}".format(start, end))

        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        self._network.update_fc(task_size)
        logging.info(
            "Task {} | known={} -> total={} | classes={}...".format(
                self._cur_task,
                self._known_classes,
                self._total_classes,
                task_classes,
            )
        )

        train_set = data_manager.get_dataset(task_classes, source="train", mode="train")
        test_set = data_manager.get_dataset(seen_classes, source="test", mode="test")
        logging.info(
            "Task {} | train size={} test size={}".format(
                self._cur_task,
                len(train_set),
                len(test_set),
            )
        )

        # Train on current task
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self._stage1_training(train_loader, test_loader)
        stage1_acc = self._evaluate(test_loader) * 100.0
        logging.info("Task {} post-stage1 top1 {:.3f}".format(self._cur_task, stage1_acc))

        if hasattr(self._network, "fc") and hasattr(self._network.fc, "backup"):
            self._network.fc.backup()

        # Compute Fisher information if enabled
        if self.fisher_weighting:
            self.fisher_mat.append(self._get_fisher_diagonal(train_loader, self.optimizer))

        # Apply model averaging
        if self._cur_task > 0:
            theta_0 = self.prev_nets[self.init_w]
            theta_1 = self._snapshot_state_dict()
            theta = self._interpolate_weights(
                theta_0,
                theta_1,
                alpha=self.wt_alpha,
                fisher=self.fisher_weighting,
                fisher_mat=self.fisher_mat[-2:] if self.fisher_weighting else None,
            )
            self._load_state_dict(theta)
            logging.info(
                "Model averaging applied | task={} alpha={:.3f} fisher={}".format(
                    self._cur_task, self.wt_alpha, self.fisher_weighting
                )
            )
            interp_acc = self._evaluate(test_loader) * 100.0
            logging.info("Task {} post-interp top1 {:.3f}".format(self._cur_task, interp_acc))

        # Compact classifier
        self._compute_class_mean(data_manager)

        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage2_compact_classifier(test_loader, task_size)
            ca_acc = self._evaluate(test_loader) * 100.0
            logging.info("Task {} post-ca top1 {:.3f}".format(self._cur_task, ca_acc))

        acc = self._evaluate(test_loader)
        top1 = acc * 100.0
        logging.info("Task {} top1 {:.3f}".format(self._cur_task, top1))
        return top1

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        ce = nn.CrossEntropyLoss()
        for epoch in range(self.epochs):
            self._network.train()
            epoch_loss = 0.0
            batch_count = 0
            for _, inputs, targets in train_loader:
                outputs = self._network(inputs, bcb_no_grad=self.fix_bcb)
                cur_targets = targets - self._known_classes
                logits = outputs["logits"][:, self._known_classes : self._total_classes]
                loss = ce(logits, cur_targets)
                optimizer.step(loss)
                epoch_loss += float(loss.item())
                batch_count += 1

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(1, batch_count)
            epoch_id = epoch + 1
            if epoch_id % 5 == 0 or epoch_id == self.epochs:
                train_acc = self._evaluate(train_loader)
                test_acc = self._evaluate(test_loader)
                logging.info(
                    "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}".format(
                        self._cur_task,
                        epoch_id,
                        self.epochs,
                        avg_loss,
                        train_acc * 100.0,
                        test_acc * 100.0,
                    )
                )
            else:
                logging.info(
                    "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                        self._cur_task,
                        epoch_id,
                        self.epochs,
                        avg_loss,
                    )
                )

    def _stage1_training(self, train_loader, test_loader):
        base_params = self._network.convnet.parameters()
        head_params = list(self._network.fc.parameters())

        if not self.fix_bcb:
            for param in self._network.convnet.parameters():
                try:
                    param.start_grad()
                except Exception:
                    pass
            network_params = [
                {
                    "params": base_params,
                    "lr": self.lr * self.bcb_lrscale,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": head_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                },
            ]
        else:
            for param in self._network.convnet.parameters():
                try:
                    param.stop_grad()
                except Exception:
                    try:
                        param.requires_grad = False
                    except Exception:
                        pass
            network_params = [
                {
                    "params": head_params,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                }
            ]

        try:
            optimizer = jt.optim.SGD(
                network_params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        except Exception:
            optimizer = jt.optim.SGD(
                self._network.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )

        scheduler = None
        try:
            scheduler = jt.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.milestones,
                gamma=self.lrate_decay,
            )
        except Exception:
            scheduler = None

        self.optimizer = optimizer
        self._run(train_loader, test_loader, optimizer, scheduler)

    def _evaluate(self, data_loader):
        total = 0
        correct = 0
        self._network.eval()
        with jt.no_grad():
            for _, inputs, targets in data_loader:
                if isinstance(targets, (tuple, list)):
                    targets = targets[0]
                if not isinstance(targets, jt.Var):
                    targets = jt.array(targets)
                targets = targets.int32()
                outputs = self._network(inputs)
                preds = jt.argmax(outputs["logits"], dim=1)[0]
                total += targets.shape[0]
                correct += int((preds == targets).sum())
        return correct / max(1, total)

    def _compute_class_mean(self, data_manager):
        feat_dim = self._network.feature_dim
        if self._class_means is not None:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_means = np.zeros((self._total_classes, feat_dim), dtype="float32")
            new_means[: self._known_classes] = self._class_means
            self._class_means = new_means

            new_covs = np.zeros((self._total_classes, feat_dim, feat_dim), dtype="float32")
            new_covs[: self._known_classes] = self._class_covs
            self._class_covs = new_covs
        else:
            self._class_means = np.zeros((self._total_classes, feat_dim), dtype="float32")
            self._class_covs = np.zeros((self._total_classes, feat_dim, feat_dim), dtype="float32")

        self._network.eval()
        with jt.no_grad():
            for class_id in range(self._known_classes, self._total_classes):
                class_set = data_manager.get_dataset([class_id], source="train", mode="test")
                class_loader = DataLoader(
                    class_set,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )

                feats_list = []
                for _, inputs, _ in class_loader:
                    feats = self._network.convnet(inputs)
                    feats_list.append(feats.stop_grad().numpy())

                if len(feats_list) == 0:
                    continue

                feats_np = np.concatenate(feats_list, axis=0)
                mean = feats_np.mean(axis=0).astype("float32")
                if feats_np.shape[0] <= 1:
                    cov = np.eye(feat_dim, dtype="float32") * 1e-4
                else:
                    cov = np.cov(feats_np.astype("float64").T).astype("float32")
                    cov = cov + np.eye(cov.shape[0], dtype="float32") * 1e-4

                self._class_means[class_id, :] = mean
                self._class_covs[class_id, ...] = cov

    def _stage2_compact_classifier(self, test_loader, task_size):
        run_epochs = self.ca_epochs
        crct_num = self._total_classes
        num_sampled_pcls = self.ca_samples_per_class

        fc_params = list(self._network.fc.parameters())
        for param in fc_params:
            try:
                param.start_grad()
            except Exception:
                try:
                    param.requires_grad = True
                except Exception:
                    pass

        fc_optimizer = jt.optim.SGD(
            fc_params,
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )
        scheduler = None
        try:
            scheduler = jt.lr_scheduler.CosineAnnealingLR(fc_optimizer, T_max=run_epochs)
        except Exception:
            scheduler = None

        ce = nn.CrossEntropyLoss()
        self._network.eval()

        for _ in range(run_epochs):
            losses = 0.0

            sampled_data = []
            sampled_label = []

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = self._class_means[c_id] * (0.9 + decay)
                cls_cov = self._class_covs[c_id]
                cls_cov = 0.5 * (cls_cov + cls_cov.T)
                cls_cov = cls_cov + np.eye(cls_cov.shape[0], dtype="float32") * 1e-6
                sampled_data_single = np.random.multivariate_normal(
                    cls_mean,
                    cls_cov,
                    size=num_sampled_pcls,
                ).astype("float32")
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            inputs = np.concatenate(sampled_data, axis=0)
            targets = np.array(sampled_label, dtype="int32")

            sf_indexes = np.random.permutation(inputs.shape[0])
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                begin = _iter * num_sampled_pcls
                end = (_iter + 1) * num_sampled_pcls
                if begin >= inputs.shape[0]:
                    break

                inp = jt.array(inputs[begin:end])
                tgt = jt.array(targets[begin:end]).int32()
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs["logits"]

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for task_idx in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[task_idx]
                        task_logits = logits[:, prev_t_size:cur_t_size]
                        temp_norm = jt.sqrt((task_logits * task_logits).sum(dim=-1, keepdims=True) + 1e-7)
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[task_idx]

                    per_task_norm = jt.concat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdims=True)
                    decoupled_logits = (logits[:, :crct_num] / norms) / self.logit_norm
                    loss = ce(decoupled_logits, tgt)
                else:
                    loss = ce(logits[:, :crct_num], tgt)

                fc_optimizer.step(loss)
                losses += float(loss.item())

            if scheduler is not None:
                scheduler.step()

            test_acc = self._evaluate(test_loader)
            logging.info(
                "CA Task {} => Loss {:.3f}, Test_accy {:.3f}".format(
                    self._cur_task,
                    losses / max(1, crct_num),
                    test_acc * 100.0,
                )
            )

    def after_task(self):
        self.prev_nets.append(self._snapshot_state_dict())
        self._known_classes = self._total_classes
        if hasattr(self._network, "fc") and hasattr(self._network.fc, "recall"):
            self._network.fc.recall()
        logging.info(
            "Task {} finished | known_classes={}".format(self._cur_task, self._known_classes)
        )

    def _snapshot_state_dict(self):
        state = self._network.state_dict()
        snap = {}
        for key, value in state.items():
            snap[key] = value.stop_grad().numpy().copy()
        return snap

    def _load_state_dict(self, state):
        try:
            self._network.load_state_dict(state, strict=True)
        except TypeError:
            self._network.load_state_dict(state)

    def _interpolate_weights(self, theta_0, theta_1, alpha, fisher=False, fisher_mat=None):
        if not fisher:
            theta = {
                key: ((1 - alpha) * jt.array(theta_0[key]) + alpha * jt.array(theta_1[key])).stop_grad()
                for key in theta_0.keys()
            }
            unique_keys = set(theta_1.keys()) - set(theta_0.keys())
            for key in unique_keys:
                theta[key] = jt.array(theta_1[key]).stop_grad()
            return theta

        if fisher_mat is None or len(fisher_mat) != 2:
            raise ValueError("Fisher-weighted interpolation requires fisher_mat of length 2.")

        fisher_0, fisher_1 = fisher_mat
        F_theta1 = {
            key: jt.array(fisher_1[key]) * jt.array(theta_1[key])
            for key in theta_1.keys()
        }
        F_theta0 = {
            key: jt.array(fisher_0[key]) * jt.array(theta_0[key])
            for key in theta_0.keys()
        }

        theta = {
            key: (
                ((1 - alpha) * F_theta0[key] + alpha * F_theta1[key])
                / ((1 - alpha) * jt.array(fisher_0[key]) + alpha * jt.array(fisher_1[key]))
            ).stop_grad()
            for key in theta_0.keys()
        }

        unique_keys = set(theta_1.keys()) - set(theta_0.keys())
        for key in unique_keys:
            theta[key] = F_theta1[key].stop_grad()
        return theta

    def _get_fisher_diagonal(self, train_loader, optimizer):
        fisher = {}
        named_params = list(self._network.named_parameters())
        for name, param in named_params:
            fisher[name] = jt.zeros_like(param).stop_grad()

        self._network.train()
        for _, inputs, targets in train_loader:
            outputs = self._network(inputs, bcb_no_grad=self.fix_bcb)
            loss = nn.cross_entropy_loss(outputs["logits"], targets, reduction="mean")

            if optimizer is not None and hasattr(optimizer, "zero_grad"):
                optimizer.zero_grad()

            grads = jt.grad(loss, [param for _, param in named_params])
            for (name, _), grad in zip(named_params, grads):
                if grad is not None:
                    fisher[name] += grad.stop_grad() * grad.stop_grad()

            if self.manual_release:
                jt.sync_all()
                jt.gc()

        steps = len(train_loader)
        for name in fisher:
            fisher[name] = jt.minimum(
                fisher[name] / max(1, steps),
                jt.ones_like(fisher[name]) * self.fisher_max,
            ).stop_grad()
        return fisher